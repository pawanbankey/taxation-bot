import faiss
import numpy as np
import json, re, os, requests, uuid, threading
from openai import AzureOpenAI
import matplotlib.pyplot as plt
import io, base64, pytz
from io import BytesIO
from PIL import Image
import logging
from azure.storage.blob import BlobServiceClient
import tempfile
import platform
from datetime import datetime
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
import json
from pymongo import UpdateOne
import pymongo
import logging
from pymongo import MongoClient
import unicodedata

def download_specific_blob(blob_name):
    connection_string = "DefaultEndpointsProtocol="
    container_name = ""

    if not container_name:
        raise ValueError("Please specify a container name.")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string, connection_verify=False)
    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(blob_name)

    # download_dir = '/tmp/finbotFiles'
    
    if platform.system() == 'Windows':
        download_dir = os.path.join(tempfile.gettempdir(), 'finbotFiles')
    else:
        download_dir = '/tmp/finbotFiles'
    os.makedirs(download_dir, exist_ok=True)

    file_path = os.path.join(download_dir, blob_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as file:
        file.write(blob_client.download_blob().readall())

    logging.info(f"finbot : Downloaded {blob_name} to {file_path}")
    return file_path

# Azure OpenAI setup
embedding_model = "text-embedding-ada-002"
client = AzureOpenAI(
    api_key="",  # Replace with your actual key or use environment variable
    azure_endpoint="",
    api_version=""
)

# Load FAISS index


# update mongo
def update_or_add_field(convo_id, update_data):
    try:
        """
        Updates or adds a new field to documents in a MongoDB collection.

        Parameters:
        - db_name (str): Name of the database.
        - collection_name (str): Name of the collection.
        - query_filter (dict): Filter to select documents to update.
        - new_field (str): Name of the new field to add or update.
        - new_value: Value to set for the new field.
        """
        client = MongoClient("mongodb+srv://AIML_DB_UAT_USR:2xiQ8K5cpfmTpCS8@poonawallacluster0.to3tz.mongodb.net/")  # Adjust URI as needed
        db = client['AIML_DB_UAT']
        collection = db['finbot']

        update_data_with_id = {"convo_id": convo_id, **update_data}

        result = collection.update_one(
            {"convo_id": convo_id},
            {"$set": update_data_with_id},
            upsert=True
        )
        return {"status": "Sucesses", "result": str(result)}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}

# Function to embed a query
def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=embedding_model
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
    return query_embedding

def send_message(convo_id, email, query, response, graph_generated, image_base64):
    url = "https://ee3ade3a7d2ae73ca7329979494f18.d4.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/d474e22526fd4b51a6f6e38252b5bf72/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=6RTOGn-nkL-0DQOWBAGoQbRfooti0GKDOadtCwfTfoY"

    payload = json.dumps({
    "convo_id": str(convo_id),
    "email": str(email),
    "query": str(query),
    "response": str(response),
    "graph_generated": str(graph_generated),
    "image_base64":str(image_base64)
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload, verify=False)

    # print(response.text)
    return response.text

def search_index(search_keywords, top_k=10):
    """
    Query Azure Cognitive Search and return top `top_k` chunks with search score.
    Optionally filter using OData filter syntax if `filter_expr` is provided.
    """

    # Azure Search parameters
    search_endpoint = ""
    search_api_key = ""
    index_name = "rag-finance-faqs-updates" #"rag-finance-faqs" #"financebot"

    if not search_api_key or "REDACTED" in search_api_key:
        logging.warning("AZURE_SEARCH_API_KEY not set. Using placeholder. Set env var for security.")

    headers = {
        'Content-Type': 'application/json',
        'api-key': search_api_key
    }

    # Use your current API version; consider upgrading to a recent stable when you can.
    url = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2020-06-30"

    body = {
        "search": search_keywords,
        "top": top_k,
        "select": "chunk,source",
        "orderby": "search.score() desc",
        # "highlight": "content_text",
        # "queryType": "full",
        # "searchMode": "all"
    }

    logging.info(f"finbot : Searching for policies using keywords: {search_keywords} (top={top_k})")
    resp = requests.post(url, headers=headers, json=body, verify=False)

    if not resp.ok:
        logging.error(f"Azure Search error {resp.status_code}: {resp.text}")
        return 'error'

    payload = resp.json()
    docs = payload.get('value', [])

    # Defensive client-side sort in case orderby isn't honored
    docs.sort(key=lambda d: d.get('@search.score', 0.0), reverse=True)

    results = []
    for d in docs[:top_k]:
        # policychunk = d.get('content_text')
        policychunk = d.get('chunk')
        
        documentTitle = d.get('source')
        searchScore = d.get('@search.score', 0.0)
        # highlights = d.get('@search.highlights', {}).get('content_text', [])  # may be empty if not supported

        results.append({
            "documentTitle": documentTitle,
            "documentchunk": policychunk,
            "searchScore": searchScore,
            # Optional fields you might find useful:
            # "highlights": highlights,
            # "key": d.get("id") or d.get("metadata_storage_name")  # depends on your schema
        })
    logging.info(f"finbot : Search results: {results}")
    return results

def Agent(prompt):
    response = client.chat.completions.create(
        model="gpt-5-mini", #"gpt-4o"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        # temperature=0.3,
    )
    return response.choices[0].message.content

def generate_financial_graph(
    data, labels, title, x_label, y_label,
    graph_type='bar'
):
    """
    Generates a financial graph, saves it to disk, and/or returns a base64 PNG string.

    Parameters:
        data (list): numerical values.
        labels (list): labels for each data point.
        title (str): chart title.
        x_label (str): x-axis label.
        y_label (str): y-axis label.
        graph_type (str): 'bar', 'line', or 'pie'.
        save_path (str): if provided, path to save the PNG file.
        return_base64 (bool): if True, return the image as a base64 string.

    Returns:
        tuple: (fig, ax) by default;
        if return_base64=True, returns (fig, ax, img_base64).
    """
    # auto‐size for long labels
    width = max(8, len(labels) * 1.2)
    height = 6
    fig, ax = plt.subplots(figsize=(width, height))

    if graph_type == 'bar':
        ax.bar(labels, data, color='skyblue')
    elif graph_type == 'line':
        ax.plot(labels, data, marker='o', color='green')
    elif graph_type == 'pie':
        ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
    else:
        raise ValueError("Invalid graph_type. Choose from 'bar', 'line', or 'pie'.")

    if graph_type != 'pie':
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.setp(ax.get_xticklabels(), ha='right')
    else:
        ax.set_title(title)

    fig.tight_layout()
    img_base64 = None
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('ascii')
    buffer.close()
    plt.close(fig)

    return img_base64
    # with open ("financial_graph.txt", "wb") as f:
    #     f.write(img_base64)
    
def base64_to_image(base64_str, output_path):
    """
    Decodes a base64 string and saves it as an image file.

    Parameters:
        base64_str (str): Base64-encoded image data.
        output_path (str): File path to save the decoded image.
    """
    # If the string has a data URI scheme prefix, strip it
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img.save(output_path)

def extract_python_code(text):
    # This regex looks for code blocks enclosed in triple backticks with 'python' or without language specifier
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return code_blocks

def check_relevance(chunk_id, chunk_text, updatedQuery, companies_mentioned):
    prompt = f"""
    You are a relevance checker for a RAG system. Given a user query and a chunk, determine if the chunk is relevant and matches any of the companies mentioned.

    Query: {updatedQuery}
    Companies: {companies_mentioned}
    Chunk: {chunk_text}

    Return 'Relevant' if the chunk is useful and matches one of the companies, otherwise return 'Irrelevant'.
    """
    result = Agent(prompt)
    return (chunk_id, chunk_text) if "Relevant" in result else None

def askQuery(query):
    logs=[]
    try:        
        masterPlanningSteps = Agent(
            prompt=f""" 

                You are a finance manager in an NBFC , planning and query-composition assistant for Azure Cognitive Search in a RAG system. You are part of the finance team and have strong knowledge of financial terminology.

                TASK:

                1) Parse the user's query into granular tasks:
                - Extract company names from the provided list (Company 1 - Company 17). 
                - Identify the user query, and rewrite the query providing a give brief description about the users query also based on the task decided. Make sure to add the short form and the full form of the query in the rewritten query.
                - Identify fiscal years based on the query. If the query specifies FY, normalize it to numeric year (e.g., FY25 → 2025). or If it is not mentioned assume the latest fiscal year {datetime.today().strftime("%Y")}.

                2) For each combination of company, fiscal year, and user query, create a planning step with a query to search Azure Cognitive Search index for relevant document chunks.
                - make sure to use AND between company, year, and query clauses.
                - Use the format: "<company> | <query> | FY<year> | " followed by the Lucene query.

                3) Output strictly valid JSON.
                - it should contain the company name
                - fiscal year
                - query
                - search_keywords   

                CONSTRAINTS:
                - Do NOT add comments or extra fields.
                - Do NOT include trailing commas.
                - No newlines inside "search_keywords".
                - search_keywords must start with: "<company> | <query> | FY<year> | " followed by the Lucene query.
                - Use AND between company, year, and query clauses.

                User query: {query}
                Output the planning steps as a JSON array.
                Example output:
                [
                    {{
                        "company": "Company name",
                        "fiscal_year": 2025,
                        "query": "What is the PAT?",
                        "search_keywords": "Company name | PAT | FY2025 | "
                    }},
                    {{
                        "company": "Company name",
                        "fiscal_year": 2024,
                        "query": "What is the CRAR?",
                        "search_keywords": "Company name | CRAR | FY2024 | "
                    }}
                ]

                """
            )

        logs.append(f"masterPlanningSteps {masterPlanningSteps}")
        logging.info(f"finbot : masterPlanningSteps {masterPlanningSteps}")
        try:
            masterPlanningSteps = json.loads(masterPlanningSteps)
        except:
            masterPlanningSteps = json.loads(re.search(r'```json\n(.*?)\n```', masterPlanningSteps, re.DOTALL).group(1)) 

        finalanswers=[]

        for planM in masterPlanningSteps:
            logging.info(f"finbot : Executing plan: {planM}")
            dataChunks = search_index(planM['search_keywords'])
            agent1=[]
            for dataC in dataChunks:
                dataVerify=Agent(
                    prompt=f""" 

                        You are a precise information extractor and a financial expert.
                        Extract query value from ONE text chunk for a given company and fiscal year (FY) (if it is mentioned in the planning step). 
                        
                        Rules:
                        - Use ONLY the provided chunk; do not infer or invent values.
                        - If we can derive the value using calculation then derive the value and so provide the calculation.
                        - Do verify the financial year mentioned in the chunk matches the fiscal year in the planning step.
                        - Do verify the company name against the document title, so that you are using the correct company's document to find the answer.
                        - Evidence: return the exact substring that supports the value along with the metric (20-200 chars) and character span indexes if possible.
                        - Output STRICT JSON only. No extra commentary.
                            - company names
                            - fiscal_year
                            - query
                            - calculations if needed to derive the value
                            - value: answer or null if not found
                            - documentTitle : source document title name


                        planningStep: {planM}, 

                        datachunk : {dataC}

                    """
                )
                try:
                    dataVerify = json.loads(re.search(r'```json\n(.*?)\n```', dataVerify, re.DOTALL).group(1))
                except:
                    dataVerify = json.loads(dataVerify) 
                agent1.append(dataVerify)
            
            
            logs.append(f"agent1 dataVerify for plan {planM}")
            logs.append(f"Data Chunk being processed {dataC}")
            logs.append(f"agent1 dataVerify output {agent1}")
            
            finalAns=Agent(
                prompt=f"""
                    You are a financial expert and manager along with reconciliation engine for financial KPIs.
                    Given multiple extraction data (JSON objects) for the SAME company, fiscal year, and metric:
                    - Consider only data with present=true and fy_match=true; prefer company_match=true.
                    - Prefer higher search_score and authoritative sources (annual reports, investor presentations if its present).
                    - If >=2 data agree within ±5%, accept the consensus.
                    - If values disagree, pick the most credible and report a range in notes.
                    - If no valid data, return not_found=true.


                    Output STRICT JSON only:
                    it should contain
                    - "company": string,
                    - "fiscal_year": number,
                    - "query": string,
                    - "answer" : string
                    - "documentTitle" : string
                    - "calculations": 
                    - "confidence": number,  // 0.0–1.0
                    - "not_found": boolean,
                    - "decision": "consensus" | "best_source" | "conflict" | "insufficient",
                    - "citations": [
                        {{ "doc_id": string, "doc_title": string, "evidence": string, "search_score": number | null , "doc_page": number | null }}
                    ],
                    "notes": string | null
                

                    agent1 output: {agent1}

                """
            )
            try:
                finalAns = json.loads(re.search(r'```json\n(.*?)\n```', finalAns, re.DOTALL).group(1))
            except:
                finalAns = json.loads(finalAns)

            finalanswers.append(finalAns)

        logging.info(f"finbot : Exceuted the plans and the response is {finalanswers}")
        logs.append(f"Exceuted the plans and the response is {finalanswers}")
        
        answer=Agent(
            prompt=f""" 
                You are a financial expert and manager. Your job is to convert structured KPI extraction results into a clear, concise, and user-friendly answer.

                RULES:
                - Use ONLY the provided JSON result.
                - State the company, user query, and the answer.
                - If confidence < 0.7 or not_found=true, say "Data not found in available documents."
                - If confidence ≥ 0.7, provide the value and a short explanation.
                - Include citations in plain text: mention the document title(s) and optionally the evidence snippet along with page number.
                - Do NOT invent or infer any data beyond what is in the JSON.
                - Keep the tone professional and factual.
                - add confidence in the answer.
                - if any vlaue is being there, try to give the number in INR if possible or give as it is.
                - remember you are giving answers to the finance experts, so frame the answers accordingly.
                - strictly add the citations in the answer.

                JSON results: {finalanswers}

            """
        )

        logging.info(f"finbot : Final answer {answer}")
        logs.append(f"Final answer {answer}")
        return answer, logs
    except Exception as e:
        logging.error(f"Error in askQuery: {str(e)}")
        logs.append(f"Error in askQuery: {str(e)}")
        return f"Just got hit by a error!! can you please retry asking the question!!", logs

def sanitize_string(input_str, target_encoding='ascii', replace_with='?'):
    # Normalize Unicode characters (e.g., accented letters)
    normalized = unicodedata.normalize('NFKD', input_str)
    
    # Encode to target encoding, replacing unsupported characters
    encoded = normalized.encode(target_encoding, errors='replace')
    
    # Decode back to string
    return encoded.decode(target_encoding)

def finbotQuery(convoID, employee_email, query):
    finbotQueryLogs=[]
    try:
        logging.info("finbot : finbotQuery function started")
        finbotQueryLogs.append("finbotQuery function started")
        finbotQueryLogs.append(f"convoID: {convoID}, employee_email: {employee_email}, query: {query}")

        classify=call_ai(
            prompt=f"""
                You are a helpful assistant that classifies user queries into 'query' or 'presentation' based on the following criteria:

                query:
                - Direct questions seeking specific financial data or explanations.

                presentation:
                - requires presentation of a specific company.
                - the user is asking for a presentation or a report format.
                - the word 'presenation' is mentioned in the query.
                
                Classify the following query: "{query}"

                Respond with either "query" or "presentation" only.
                The reponse should be in json format as:
                {{
                    "classification": "query" | "presentation"
                }}               
            """,
            role="You are the expert in financial data and numbers."
        )
        logging.info(f"finbot : Classify output: {classify}")
        finbotQueryLogs.append(f"Classify output: {classify}")
        if json.loads(classify)["classification"].strip().lower() == 'presentation':
            logging.info("finbot : finbot : Presentation type query identified.")
            finbotQueryLogs.append("Presentation type query identified.")
            answer=presentation(query)
            send_message(
                convo_id=convoID, 
                email=employee_email,
                query=query,
                response =answer,
                graph_generated = '',
                image_base64=''
            )
            # return {"response":answer}
            return {"status":"Sucesses", "answer": "Done"}
        else:
            logging.info(f"finbot : convoID: {convoID}, employee_email: {employee_email}, query: {query}")
            datetime_ =datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d/%m/%Y %H:%M")
            
            update_or_add_field(convo_id=convoID, update_data=
                        {"employee_email":employee_email,
                        "query": query, 
                        "start_dateTime":datetime_,
                        "satisfaction":None,
                        })
            
            answer, logs = askQuery(query)
            
            finbotQueryLogs = finbotQueryLogs + logs
            logging.info(f"finbot : Answer from askQuery: {answer}")

            graphAgent = Agent(
                prompt=f"""
                You are an expert in creating financial graphs. Based on the expert's answer and accompanying table below (if any graph can be formed from the tables, it should contain numbers for generating a graph), generate a Python code snippet that calls the provided function:

                    Answer: {answer}

                    Function signature:
                    def generate_financial_graph(data, labels, title, x_label, y_label, graph_type='bar')

                    Parameters:
                    data       list of numeric values  
                    labels     list of labels matching each data point  
                    title      chart title (string)  
                    x_label    x-axis label (string)  
                    y_label    y-axis label (string)  
                    graph_type one of 'bar', 'line', or 'pie'  

                    
                    Important Notes: 
                    - Please make sure to choose the data clearly so that the graph should help the finance team 
                    Return only the code snippet in this exact form:

                    result = generate_financial_graph(
                        data=...,
                        labels=...,
                        title='...',
                        x_label='...',
                        y_label='...',
                        graph_type='...'
                    )

                    If no table data can be formed, output exactly:

                    "no tables can be formed"

                    The output should be in json format
                    {{
                        "code_snippet": "..."
                    }}

                    """
            )
            logging.info(f"finbot : Graph agent output: {graphAgent}")

            finbotQueryLogs.append(f"Graph agent output: {graphAgent}")

            if "no tables can be formed" not in graphAgent.lower():
                code_snippet = str(graphAgent['code_snippet']).strip()
                extractedCode = extract_python_code(code_snippet)[0]
                local_vars={}
                exec(extractedCode, globals(), local_vars)
                img_base64 = local_vars['result']
                graph=True

            else:
                img_base64 = 'no tables can be formed'
                graph=False  
            
            datetime_ =datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d/%m/%Y %H:%M")

            update_or_add_field(convo_id=convoID, update_data=
                        {"employee_email":employee_email,
                        "query": query, 
                        "response":answer, 
                        "graph_generated":graph, 
                        "end_dateTime":datetime_,
                        "satisfaction":None,
                        "image_base64":img_base64, 
                        "logs":finbotQueryLogs
                        })
            
            send_message(
                convo_id=convoID, 
                email=employee_email,
                query=query,
                response =answer,
                graph_generated = graph,
                image_base64=img_base64
            )

            # return {"status":"Sucesses","answer":answer, "img_base64":img_base64}
            return {"status":"Sucesses", "answer": "Done"}
        
    except Exception as e:
        logging.error(f"finbot : Error in finbotQuery: {str(e)}")
        send_message(
            convo_id=convoID, 
            email=employee_email,
            query=query,
            response =f' Just got hit by a error!! can you please retry asking the question!! error {e}',
            graph_generated = '',
            image_base64=''
        )

        return {"status":"Failed","error Occured":str(e)}

# connected uat to the dev 
def main_FinBotQuery(convoID, employee_email, query):
    # threading.Thread(target=finbotQuery_UAT, args=(convoID, employee_email, query)).start()
    r=finbotQuery_UAT(convoID, employee_email, query)
    return {"status": "Sucesses", "answer": "Your request is being processing."}

def Agent_UAT(prompt):
    logging.info(f"finbot : Agent_UAT prompt: {prompt}")
    try:
        response = client.chat.completions.create(
            # model="gpt-4o",
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0,
            response_format={"type": "json_object"}
        )
        logging.info(f"finbot : Agent_UAT response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"finbot : Error in Agent_UAT: {str(e)}")
        return f"Error: {str(e)}"

def search_index_UAT(search_keywords, top_k=20):
    """
    Query Azure Cognitive Search and return top `top_k` chunks with search score.
    Optionally filter using OData filter syntax if `filter_expr` is provided.
    """

    # Azure Search parameters
    search_endpoint = ""
    search_api_key = ""
    index_name = "finance-rag-faq" #"rag-finance-faqs-updates" #"rag-finance-faqs-updated-indexer" #"rag-finbot-faqs" #"financebot"

    if not search_api_key or "REDACTED" in search_api_key:
        logging.warning("AZURE_SEARCH_API_KEY not set. Using placeholder. Set env var for security.")

    headers = {
        'Content-Type': 'application/json',
        'api-key': search_api_key
    }

    # Use your current API version; consider upgrading to a recent stable when you can.
    url = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2025-08-01-preview"

    body = {
        "count": True,
        "vectorQueries": [
            {
                "kind": "text",
                "text": search_keywords,
                "fields": "text_vector"
            }
        ]
    }
    # body = {
    #     "search": search_keywords,
    #     "top": top_k,
    #     "select": "chunk,source",
    #     "orderby": "search.score() desc",
    # }

    logging.info(f"finbot : Searching for policies using keywords: {search_keywords} (top={top_k})")
    resp = requests.post(url, headers=headers, json=body, verify=False)

    if not resp.ok:
        logging.error(f"Azure Search error {resp.status_code}: {resp.text}")
        return 'error'

    payload = resp.json()
    docs = payload.get('value', [])

    # Defensive client-side sort in case orderby isn't honored
    # docs.sort(key=lambda d: d.get('@search.score', 0.0), reverse=True)

    results = []
    for d in docs[:top_k]:
        searchScore = d.get('@search.score', 0.0)
        results.append({
            "company":str(d.get('company_slug', '')).replace("\u20b9", 'INR '),
            "quarter":str(d.get('quarter_slug', '')).replace("\u20b9", 'INR '),
            "fiscal_year":str(d.get('fiscal_year', '')).replace("\u20b9", 'INR '),
            "report_type":str(d.get('report_type', '')).replace("\u20b9", 'INR '),
            "page_number":str(d.get('page_number', '')).replace("\u20b9", 'INR '),
            "source":str(d.get('source', '')).replace("\u20b9", 'INR '),
            "searchScore": searchScore,
            "questionAnswer":str(d.get('chunk', '')).replace("\u20b9", 'INR ')
            # Optional fields you might find useful:
            # "highlights": highlights,
            # "key": d.get("id") or d.get("metadata_storage_name")  # depends on your schema
        })
    logging.info(f"finbot : Search results: {results}")
    return results

# def askQuery_UAT(convoID, employee_email, query):
#     logs=[]
#     try:
#         masterPlanningSteps = Agent_UAT(
#             prompt=f""" 

#                 You are a finance manager in an NBFC , planning and query-composition assistant for a RAG system. You are part of the finance team and have strong knowledge of financial terminology.

#                 Given a user query, perform the following steps to generate structured and granular tasks:

#                 1. **Extract Company Names**:
#                 - Identify and extract company names from a predefined list (e.g., Company 1 to Company 17).
#                 - If multiple companies are mentioned, treat each as a separate task.
#                 - try to give the full form of the company name if possible. (L&T Finance -> Larsen & Toubro Finance)

#                 2. **Break Down Metrics**:
#                 - If the query includes multiple metrics (e.g., "PAT and CRAR"), split them into individual sub-queries.
#                 - For each metric, include both the **short form** and the **full form** in the rewritten query.
#                     - Example: "PAT" becomes "PAT (Profit After Tax)"
#                     - Example: "CRAR" becomes "CRAR (Capital to Risk-weighted Assets Ratio)"

#                 3. **Identify and Normalize Fiscal Year**:
#                 - If the query specifies a fiscal year (e.g., "FY25", "FY2024-25"), normalize it to the format "FY2024-25".
#                 - If no fiscal year is mentioned, assume the latest fiscal year as `FY {datetime.today().strftime("%Y")}`.

#                 4. **Rewrite the User Query**:
#                 - For each combination of company, metric, and fiscal year, rewrite the query to include:
#                     - The metric with both short form and full form.
#                     - The company name.
#                     - The normalized fiscal year.
#                 - Also, provide a brief description of the user's intent based on the parsed query.

#                 5. **Output Format**:
#                 - Return the result in **strictly valid JSON** format.
#                 - Each task should include:
#                     - `company`: Name of the company.
#                     - `fiscal_year`: Normalized fiscal year.
#                     - `query`: Rewritten query with metric short form and full form, and fiscal year.
#                     - `report`: Type of report (e.g., "annual", "quarterly") if specified or inferred.

#                 **Constraints**:
#                 - Do NOT include comments or extra fields.
#                 - Do NOT include trailing commas.
#                 - Ensure all queries are split and rewritten clearly and independently.

#                 Query: {query}

#                 **Example Input**:
#                 User query: "What is the PAT and CRAR for Bajaj FY25?"

#                 **Expected Output**:
#                 ```json
#                 {{
#                 "tasks": [
#                     {{
#                     "company": "Bajaj",
#                     "report" : "anual"
#                     "fiscal_year": "FY2024-25",
#                     "query": "What is the PAT (Profit After Tax) for Bajaj in FY2024-25?"
#                     }},
#                     {{
#                     "company": "Bajaj",
#                     "fiscal_year": "FY2024-25",
#                     "query": "What is CRAR (Capital to Risk-weighted Assets Ratio) for Bajaj in FY2024-25?"
#                     }}
#                 ]
#                 }}
#                 """
#         )

#         masterPlanningSteps = json.loads(masterPlanningSteps)
    
#         logs.append(f"Master Planning Steps:")
#         logs.append(masterPlanningSteps)
        
#         Updatequeryuser=f"Hey\n Processing your query : {query}\n Please wait for few seconds, I will get back to you with the response shortly."
#         send_message(
#             convo_id=convoID, 
#             email=employee_email,
#             query=query,
#             response =Updatequeryuser,
#             graph_generated = '',
#             image_base64=''
#         )

#         slav1_logs=[]
#         slave2_logs=[]
#         masterSlaveAnalysis_logs=[]

#         for planM in masterPlanningSteps['tasks']:
#             logging.info(f"Finbot : Planning the step is {planM}")
#             logs.append(f"Planning Steps:")
#             logs.append(planM)
            
#             dataChunks = search_index_UAT(planM['query'])
#             # dataChunks = search_index_UAT(planM['query'],filter_expr="company_name eq 'Shriram Finance'")
#             logging.info(dataChunks)

#             logs.append(f"Data Chunks Retrieved:")
#             logs.append(dataChunks)

#             u_dataChunks=[]
#             for dt in dataChunks:
#                 if (planM['company'].lower() in dt['company'].lower().replace('-', ' ')):
#                     u_dataChunks.append(dt)
#             if len(u_dataChunks)<4:
#                 u_dataChunks=dataChunks
            

#             logs.append(f"after filtration chunks Retrieved:")
#             logs.append(u_dataChunks)

#             for datac in u_dataChunks:

#                 slave1=Agent_UAT(
#                     prompt = f""" 
#                         You are a precise information extractor and a financial expert.
#                         Extract the answer form the users query: {planM.get("query")}, 
                        
#                         given the dataChunk: {datac}.

#                         Rules:
#                         - Use ONLY the provided chunk; do not infer or invent values.
#                         - If we can derive the value using calculation then derive the value and so provide the calculation.
#                         - From the given chunk, there are fields like 
#                             -- company_name :<helps to verify the company name>,
#                             -- fiscal_year:<helps to verify the fiscal year it can be in the form of FY2024-25 or 2024 etc>,
#                             -- quarter : <helps to idenify the quarter of reporting, example: q1, q2, .. , annual report>
#                             -- source:<helps to verify the source document>,
#                             -- report_type:<helps to verify the type of report annual report, investor presentation etc>
#                             -- page_number:<helps to verify the page number of the document from where the chunk is taken>
#                             -- chunk:<the actual text chunk from which the answer has to be extracted>
#                             -- Try to avoid symbols and special characters in the answer.

#                         - Think logically and consider all of the above fields and extract the answer strictly in json format.
#                             -- confidence_level : <confidence level for the source (0 -100)%>
#                             -- company names <name of the company>
#                             -- fiscal_year <fiscal year mentioned in the chunk>
#                             -- query <keep the query same as the user query>
#                             -- calculations if needed to derive the value
#                             -- value: answer or null (if you are not able to find the answer in the chunk)
#                             -- source_document <source document title name>
#                             -- page_number <page number of the document from where the chunk is taken>
#                             -- logical_reasoning  <step by step reasoning for arriving at the answer>

#                     """
#                 )

#                 slave1 = json.loads(slave1)
#                 slav1_logs.append(slave1)
                
#                 logs.append(f"Slave 1 Output:")
#                 logs.append(slave1)

#                 print(f"slave1 is output {slave1}")

#                 slave2=Agent_UAT(
#                     prompt=f""" 
#                         You are a precise information verifier and a financial expert.

#                         here is the task for you:
#                         - here is the question posed to expert1 and answer provided by them
#                         --  'query': {slave1.get("query", '')},
#                         --  'value': '{slave1.get("value", '')}'.

#                         here are the data chunks to verify the answer:
#                         -- dataChunks: {dataChunks}

#                         dataChunks have fields like 
#                             -- company names :<helps to verify the company name>,
#                             -- fiscal_years:<helps to verify the fiscal year it can be in the form of FY2024-25 or 2024 etc>,
#                             -- source:<helps to verify the source document>,
#                             -- report_type:<helps to verify the type of report annual report, investor presentation etc>
#                             -- page_number:<helps to verify the page number of the document from where the chunk is taken>
#                             -- chunk:<the actual text chunk from which the answer has to be extracted>
#                             -- Try to avoid symbols and special characters in the answer.

#                         Analyze the query and the value and check the following
#                             - what will the source of the value, if the value is not null.
#                             - if the value is null, then can the value be found in any of the data chunks provided.
#                             - is there any issues with the answer provided by expert1.
#                             - Do verify the financial year and the quatar for giving the answer.


#                         Please provide the answer in the following JSON format only:
#                         {{
#                             "verified_answer": {{
#                                 "query": "{slave1.get("query", 'None')}", (verbatim as the query analysed by expert1)
#                                 "value": "{slave1.get("value", 'None')}" (verbatim as the value provided by expert1),
#                                 "is_verified": true | false (if the same value provided can be derived from the dataChunks),
#                                 "source_document": "source if verified else null",
#                                 "page_number": "Page number if verified else null",
#                                 "confidence_level": "Confidence level of verification (0-100)%" (based on how many chunks support the answer),
#                                 "verification_reasoning": "Step by step reasoning for verification"
#                                 "no_of_sources_verified": "Number of sources that support the answer"
#                             }}
#                         }}

#                     """
#                 )

#                 slave2 = json.loads(slave2)

#                 slave2_logs.append(slave2)

#                 logs.append(f"Slave 2 Output:")
#                 logs.append(slave2)

#                 logging.info(f"finbot : slave2 output {slave2}")

#                 masterSlaveAnalysis=Agent_UAT(
#                     prompt=f"""
#                         You are the financial expert.
                        
#                         User has asked the query : {planM.get("query")}

#                         there are 2 reports form differnet experts 
#                         - expert1 has provided the following detials based on the query.
#                         - expert2 has given the sources for the asnwers given by the expert1 along with any issue. 

#                         Analyze both the reports and check the following
#                         - Sources provided by both the experts should match.
#                         - If there is any mismatch in the sources provided by both the experts, then flag it.
#                         - If the issue is not ignorable then flag that.
#                         - Try to avoid symbols and special characters in the answer.

#                         expert1 output : {slave1}
                        
#                         expert2 output : {slave2}

#                         Provide the output in the following json format
                        
#                         {{
#                             "query": {planM.get("query")},
#                             "answer": <after the analysis provide the final answer here>,
#                             "reason": <reason for the final answer provided>,
#                             "flag_source_mismatch": <true/false>,
#                             "flag_critical_issue": <true/false>,
#                             "verified_sources": <names of the source document if verified else null>,
#                             "verified_page_numbers": <page numbers if verified else null>
#                         }}

#                     """
#                 )
#                 masterSlaveAnalysis = json.loads(masterSlaveAnalysis)
#                 masterSlaveAnalysis_logs.append(masterSlaveAnalysis)

#                 logs.append(f"masterSlaveAnalysis Output:")
#                 logs.append(masterSlaveAnalysis)
                
#                 logging.info(f"finbot : masterSlaveAnalysis output {masterSlaveAnalysis}")
        

#         finalAnswer=json.loads(Agent_UAT(
#             prompt=f"""
#                 You are the financial expert.
#                 Your task is to compile a final report summarizing findings from multiple analysis reports provided by different experts.

#                 Here are multiple analysis reports conducted by different experts:
                
#                 ExpertReports : {masterSlaveAnalysis_logs}
                
#                 Your task is to compile a final report summarizing the findings.

#                 Rules:
#                 - For each analysis report, extract the query and the final answer.
#                 - If either `flag_source_mismatch` or `flag_critical_issue` is true, then also consider the reason for it."
#                 - use the `answer` field as the final answer. The verified answer are more reliable.
#                 - Include citations only if the final answer is not "Data not found in available documents."
#                 - Citations should include the `verified_sources` and `verified_page_numbers`. do inlcude the source as it contains the document title.
#                 - The final answer should be clear and concise.
#                 - in the final answer if multiple values are there, then give the response with both values while mentioning the exact source and page number for each value along with the confidence level.
#                 - provide the answer in such a way that its easy to understand for a finance manager.
#                 - add "This is an AI generated response based on the available documents. You may verify the response against actual document if needed" at the end of each final answer.
#                 - If you are not confident about the answer, state the reason behind it, and the throughProcesses you followed to arrive at that conclusion based on the ExpertReports.
#                 - Try to avoid symbols and special characters in the answer.

#                 Provide the final report in the following JSON format only:
#                 {{
#                     "final_report":
#                             "query": <>,
#                             "final_answer": < if you are confident about the answer provide the final answer else say "Data not found in available documents.">,
#                             "citations": <if the final_answer is found then provide the sources_used along with the page numbers>
#                             "confidence_level":"high/low/medium" (depending on the overall confidence level of the matched responses)
#                             "reason: <reason for the final answer provided if any>"
#                 }}
#             """
#         ))
        
#         logs.append(f"finalAnswer Output:")
#         logs.append(finalAnswer)
#         logging.info(f"finbot : finalAnswer output {finalAnswer}")

#         answer=Agent_UAT(
#             prompt=f"""
#                 You are a helpful assistant.

#                Here is the user query: {query}
#                 Here is the data to prepare the answer:
#                 {finalAnswer['final_report']}

#                 Provide a detailed response to the query in clear and concise language. Use only the data provided do not add/ infer any data.
#                 Make sure no point provided above is missed in the final response.
#                 - make sure to add the detailed answer 
#                 - make sure to add the detailed citations
#                 - make sure to add the confidence level
#                 - Try to avoid symbols and special characters in the answer.
                
#                 - Response to the user query in table format.
#                 - In the table just add the numeric values in the table, and rest can be in the text format.
#                 - in the end of the asnwer , add the sources used along with the page numbers in a separate table.
#                 - also add "This is an AI generated response based on the available documents. You may verify the response against actual document if needed." at the end of the asnwer.
#                 - Make sure your tables are in the markdown format.
                
#                 Give the response in json format:
#                 {{
#                     "response": <detailed response in markdown language along with the table and the citations and detailed calculations if any. >
#                 }}
#             """
#         )
        
#         answer = json.loads(answer.replace("\u20b9", "INR "))
#         logs.append(f"Answer Output to the user:")
#         logs.append(finalAnswer)
#         logging.info(f"finbot : Final answer to the user {answer['response']}")
#         return answer['response'], logs
#     except Exception as e:
#         logs.append(f"Error in askQuery: {str(e)}")
#         logging.error(f"finbot : Error in askQuery_UAT: {str(e)}")
#         return f"Just got hit by a error!! can you please retry asking the question!!", logs




import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
import threading

# --- Assumed available in your environment ---
# Agent_UAT(prompt: str) -> str  # returns JSON string
# search_index_UAT(query: str) -> List[Dict[str, Any]]
# send_message(convo_id, email, query, response, graph_generated, image_base64)

import json
import logging
import threading
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assumed available:
# Agent_UAT(prompt: str) -> str           # returns JSON string
# search_index_UAT(query: str) -> List[Dict[str, Any]]
# send_message(convo_id, email, query, response, graph_generated, image_base64)

def askQuery_UAT(convoID: str, employee_email: str, query: str) -> Tuple[str, List[Any]]:
    """
    Threaded version of askQuery_UAT with corrected input scoping and hardened JSON/prompt handling.
    Parallelizes:
      - per-task execution (each planM in masterPlanningSteps['tasks']),
      - per data-chunk chain (Slave1 -> Slave2 -> Master analysis) inside a task.
    Returns:
      (final_user_response: str, logs: List[Any])
    """
    logs: List[Any] = []
    log_lock = threading.Lock()

    # Tune as needed for your infra and rate limits
    max_workers_task = 8
    max_workers_chunk = 8
    max_chunks_per_prompt = 10  # cap chunks included in Slave-2 prompt to control prompt size

    def log_safe(*entries: Any):
        with log_lock:
            for e in entries:
                logs.append(e)

    def safe_json_loads(text: str, replace_currency: bool = False) -> Dict[str, Any]:
        try:
            if replace_currency:
                text = text.replace("\u20b9", "INR ")
            return json.loads(text)
        except Exception as ex:
            return {"_parse_error": str(ex), "_raw": text}

    def run_agent(prompt: str, *, replace_currency: bool = False, label: str = "") -> Dict[str, Any]:
        """Run Agent_UAT and parse JSON; log raw+parsed; capture errors."""
        try:
            out = Agent_UAT(prompt=prompt)
            parsed = safe_json_loads(out, replace_currency=replace_currency)
            if label:
                log_safe(f"{label} Raw Output:", out)
                log_safe(f"{label} Parsed:", parsed)
            return parsed
        except Exception as ex:
            err = {"_agent_error": str(ex), "_prompt": prompt}
            log_safe(f"Agent Error [{label}]:", err)
            return err

    def compact_chunks(chunks: List[Dict[str, Any]], limit: int = max_chunks_per_prompt,
                       keep_fields: Tuple[str, ...] = ("company", "fiscal_year", "source", "report_type", "page_number", "chunk")) -> List[Dict[str, Any]]:
        """Reduce chunk payload size for prompt safety."""
        compact = []
        for dt in (chunks or [])[:limit]:
            compact.append({k: dt.get(k) for k in keep_fields})
        return compact

    def filter_chunks(planM: Dict[str, Any], dataChunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        company = (planM.get('company') or '').lower().replace('-', ' ')
        if not company:
            return dataChunks
        u_dataChunks = [dt for dt in dataChunks
                        if company in (dt.get('company', '').lower().replace('-', ' '))]
        return u_dataChunks if len(u_dataChunks) >= 4 else dataChunks

    def process_chunk(planM: Dict[str, Any], datac: Dict[str, Any], dataChunks: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Sequential chain inside a single chunk:
            Slave 1 -> Slave 2 -> Master-Slave Analysis
        Returns:
            (master_analysis dict, chunk_logs list)
        """
        chunk_logs: List[Any] = []

        # ----- Slave 1: Extractor -----
        slave1_prompt = f""" 
            You are a precise information extractor and a financial expert.
            Extract the answer form the users query: {planM.get("query")},
            given the dataChunk: {json.dumps(datac, ensure_ascii=False)}.

            Rules:
            - Use ONLY the provided chunk; do not infer or invent values.
            - If we can derive the value using calculation then derive the value and so provide the calculation.
            - From the given chunk, there are fields like
              -- company_name :<helps to verify the company name>,
              -- fiscal_year:<helps to verify the fiscal year it can be in the form of FY2024-25 or 2024 etc>,
              -- quarter : <helps to idenify the quarter of reporting, example: q1, q2, .. , annual report>
              -- source:<helps to verify the source document>,
              -- report_type:<helps to verify the type of report annual report, investor presentation etc>
              -- page_number:<helps to verify the page number of the document from where the chunk is taken>
              -- chunk:<the actual text chunk from which the answer has to be extracted>
              -- Try to avoid symbols and special characters in the answer.
            - Think logically and consider all of the above fields and extract the answer strictly in json format.
              -- confidence_level : <confidence level for the source (0 -100)%>
              -- company names <name of the company>
              -- fiscal_year <fiscal year mentioned in the chunk>
              -- query <keep the query same as the user query>
              -- calculations if needed to derive the value
              -- value: answer or null (if you are not able to find the answer in the chunk)
              -- source_document <source document title name>
              -- page_number <page number of the document from where the chunk is taken>
              -- logical_reasoning <step by step reasoning for arriving at the answer>
        """
        slave1 = run_agent(slave1_prompt, label="Slave 1")
        chunk_logs.extend([f"Slave 1 Output:", slave1])

        # ----- Slave 2: Verifier -----
        compacted = compact_chunks(dataChunks, limit=max_chunks_per_prompt)
        try:
            chunks_for_prompt = json.dumps(compacted, ensure_ascii=False)
        except Exception:
            chunks_for_prompt = "[]"

        slave2_prompt = f""" 
            You are a precise information verifier and a financial expert.

            here is the task for you:
            - here is the question posed to expert1 and answer provided by them
            --  'query': {json.dumps(slave1.get("query", ""), ensure_ascii=False)},
            --  'value': {json.dumps(slave1.get("value", ""), ensure_ascii=False)}.

            here are the data chunks to verify the answer:
            --  dataChunks: {chunks_for_prompt}

            dataChunks have fields like
              -- company names :<helps to verify the company name>,
              -- fiscal_years:<helps to verify the fiscal year it can be in the form of FY2024-25 or 2024 etc>,
              -- source:<helps to verify the source document>,
              -- report_type:<helps to verify the type of report annual report, investor presentation etc>
              -- page_number:<helps to verify the page number of the document from where the chunk is taken>
              -- chunk:<the actual text chunk from which the answer has to be extracted>
              -- Try to avoid symbols and special characters in the answer.

            Analyze the query and the value and check the following
              - what will the source of the value, if the value is not null.
              - if the value is null, then can the value be found in any of the data chunks provided.
              - is there any issues with the answer provided by expert1.
              - Do verify the financial year and the quatar for giving the answer.

            Please provide the answer in the following JSON format only:
            {{
              "verified_answer": {{
                "query": {json.dumps(slave1.get("query", "None"), ensure_ascii=False)},
                "value": {json.dumps(slave1.get("value", "None"), ensure_ascii=False)},
                "is_verified": true,
                "source_document": "source if verified else null",
                "page_number": "Page number if verified else null",
                "confidence_level": "Confidence level of verification (0-100)%",
                "verification_reasoning": "Step by step reasoning for verification",
                "no_of_sources_verified": "Number of sources that support the answer"
              }}
            }}
        """
        slave2 = run_agent(slave2_prompt, label="Slave 2")
        chunk_logs.extend([f"Slave 2 Output:", slave2])

        # ----- Master-Slave Analysis (per-chunk synthesis) -----
        master_prompt = f"""
            You are the financial expert.

            User has asked the query : {json.dumps(planM.get("query"), ensure_ascii=False)}
            there are 2 reports form differnet experts
              - expert1 has provided the following detials based on the query.
              - expert2 has given the sources for the asnwers given by the expert1 along with any issue.

            Analyze both the reports and check the following
              - Sources provided by both the experts should match.
              - If there is any mismatch in the sources provided by both the experts, then flag it.
              - If the issue is not ignorable then flag that.
              - Try to avoid symbols and special characters in the answer.

            expert1 output : {json.dumps(slave1, ensure_ascii=False)}
            expert2 output : {json.dumps(slave2, ensure_ascii=False)}

            Provide the output in the following json format
            {{
              "query": {json.dumps(planM.get("query"), ensure_ascii=False)},
              "answer": "<after the analysis provide the final answer here>",
              "reason": "<reason for the final answer provided>",
              "flag_source_mismatch": "<true/false>",
              "flag_critical_issue": "<true/false>",
              "verified_sources": "<names of the source document if verified else null>",
              "verified_page_numbers": "<page numbers if verified else null>"
            }}
        """
        master_analysis = run_agent(master_prompt, label="Master-Slave Analysis")
        chunk_logs.extend([f"Master-Slave Analysis Output:", master_analysis])

        return master_analysis, chunk_logs

    def process_task(idx: int, planM: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]], List[Any]]:
        """
        Process a single task:
          - search_index_UAT
          - filter chunks
          - parallel process each chunk (each chunk runs the chain)
        Returns:
          (task_index, master_analysis_list, task_logs)
        """
        task_logs: List[Any] = []
        try:
            task_logs.extend([f"Planning Steps:", planM])

            # Retrieve chunks (I/O-bound)
            dataChunks = search_index_UAT(planM['query'])
            task_logs.extend([f"Data Chunks Retrieved:", dataChunks])

            # Filter chunks by company
            u_dataChunks = filter_chunks(planM, dataChunks)
            task_logs.extend([f"after filtration chunks Retrieved:", u_dataChunks])

            # Parallelize the chunk chain
            master_analysis_list: List[Dict[str, Any]] = []
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers_chunk) as chunk_executor:
                for datac in u_dataChunks:
                    futures.append(chunk_executor.submit(process_chunk, planM, datac, dataChunks))

                for fut in as_completed(futures):
                    try:
                        master_analysis, chunk_logs = fut.result()
                        # Guard: only accept dicts with a 'query' key to keep ExpertReports valid
                        if isinstance(master_analysis, dict) and master_analysis.get("query") is not None:
                            master_analysis_list.append(master_analysis)
                        else:
                            task_logs.append({"_master_analysis_invalid": master_analysis})
                        task_logs.extend(chunk_logs)
                    except Exception as ex:
                        err = {"_chunk_error": str(ex)}
                        task_logs.extend([f"Chunk Error:", err])

            return idx, master_analysis_list, task_logs

        except Exception as ex:
            task_logs.extend([f"Task Error:", {"_task_error": str(ex), "planM": planM}])
            return idx, [], task_logs

    # -------- Master planning --------
    try:
        master_prompt = f""" 
            You are a finance manager in an NBFC , planning and query-composition assistant for a RAG system. 
            You are part of the finance team and have strong knowledge of financial terminology.

            Given a user query, perform the following steps to generate structured and granular tasks:
            1. **Extract Company Names**:
               - Identify and extract company names from a predefined list (e.g., Company 1 to Company 17).
               - If multiple companies are mentioned, treat each as a separate task.
               - try to give the full form of the company name if possible. (L&T Finance -> Larsen & Toubro Finance)
            2. **Break Down Metrics**:
               - If the query includes multiple metrics (e.g., "PAT and CRAR"), split them into individual sub-queries.
               - For each metric, include both the **short form** and the **full form** in the rewritten query.
               - Example: "PAT" becomes "PAT (Profit After Tax)"
               - Example: "CRAR" becomes "CRAR (Capital to Risk-weighted Assets Ratio)"
            3. **Identify and Normalize Fiscal Year**:
               - If the query specifies a fiscal year (e.g., "FY25", "FY2024-25"), normalize it to the format "FY2024-25".
               - If no fiscal year is mentioned, assume the latest fiscal year as `FY {{datetime.today().strftime("%Y")}}`.
            4. **Rewrite the User Query**:
               - For each combination of company, metric, and fiscal year, rewrite the query to include:
                 - The metric with both short form and full form.
                 - The company name.
                 - The normalized fiscal year.
               - Also, provide a brief description of the user's intent based on the parsed query.
            5. **Output Format**:
               - Return the result in **strictly valid JSON** format.
               - Each task should include:
                 - `company`: Name of the company.
                 - `fiscal_year`: Normalized fiscal year.
                 - `query`: Rewritten query with metric short form and full form, and fiscal year.
                 - `report`: Type of report (e.g., "annual", "quarterly") if specified or inferred.
            **Constraints**:
               - Do NOT include comments or extra fields.
               - Do NOT include trailing commas.
               - Ensure all queries are split and rewritten clearly and independently.
            Query: {json.dumps(query, ensure_ascii=False)}
        """
        master_raw = Agent_UAT(prompt=master_prompt)
        masterPlanningSteps = safe_json_loads(master_raw)
        log_safe("Master Planning Steps Raw:", master_raw)
        log_safe("Master Planning Steps Parsed:", masterPlanningSteps)

        # Notify user early
        Updatequeryuser = (
            f"Hey\n Processing your query : {query}\n Please wait for few seconds, "
            f"I will get back to you with the response shortly."
        )
        try:
            send_message(
                convo_id=convoID,
                email=employee_email,
                query=query,
                response=Updatequeryuser,
                graph_generated='',
                image_base64=''
            )
        except Exception as send_ex:
            log_safe("Send Message Error:", str(send_ex))

        tasks = masterPlanningSteps.get('tasks', [])
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("No valid tasks generated in masterPlanningSteps.")

        # -------- Parallelize per task --------
        masterSlaveAnalysis_logs_all: List[List[Dict[str, Any]]] = [None] * len(tasks)
        task_logs_accum: List[List[Any]] = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=max_workers_task) as task_executor:
            futures_map = {
                task_executor.submit(process_task, idx, planM): idx
                for idx, planM in enumerate(tasks)
            }
            for fut in as_completed(futures_map):
                idx = futures_map[fut]
                try:
                    ret_idx, master_analysis_list, task_logs = fut.result()
                    masterSlaveAnalysis_logs_all[ret_idx] = master_analysis_list
                    task_logs_accum[ret_idx] = task_logs
                except Exception as ex:
                    masterSlaveAnalysis_logs_all[idx] = []
                    task_logs_accum[idx] = [f"Task Future Error:", str(ex)]

        # Merge logs in original order
        for tl in task_logs_accum:
            if tl is not None:
                log_safe(*tl)

        # Flatten master analyses as ExpertReports
        masterSlaveAnalysis_logs: List[Dict[str, Any]] = []
        for mal in masterSlaveAnalysis_logs_all:
            if mal:
                masterSlaveAnalysis_logs.extend(mal)

        log_safe("masterSlaveAnalysis Aggregate Output:", masterSlaveAnalysis_logs)

        # -------- Final report compilation --------
        if not masterSlaveAnalysis_logs:
            # Fail-open with a minimal, consistent final report if no analyses survived
            finalAnswer = {
                "final_report": {
                    "query": query,
                    "final_answer": "Data not found in available documents.",
                    "citations": [],
                    "confidence_level": "low",
                    "reason": "No successful expert analyses were available."
                }
            }
            log_safe("finalAnswer Output (fallback):", finalAnswer)
        else:
            final_prompt = f"""
                You are the financial expert.
                Your task is to compile a final report summarizing findings from multiple analysis reports provided by different experts.

                Here are multiple analysis reports conducted by different experts:
                ExpertReports : {json.dumps(masterSlaveAnalysis_logs, ensure_ascii=False)}

                Rules:
                - For each analysis report, extract the query and the final answer.
                - If either `flag_source_mismatch` or `flag_critical_issue` is true, then also consider the reason for it.
                - use the `answer` field as the final answer. The verified answer are more reliable.
                - Include citations only if the final answer is not "Data not found in available documents."
                - Citations should include the `verified_sources` and `verified_page_numbers`. do inlcude the source as it contains the document title.
                - The final answer should be clear and concise.
                - in the final answer if multiple values are there, then give the response with both values while mentioning the exact source and page number for each value along with the confidence level.
                - provide the answer in such a way that its easy to understand for a finance manager.
                - add "This is an AI generated response based on the available documents. You may verify the response against actual document if needed" at the end of each final answer.
                - If you are not confident about the answer, state the reason behind it, and the throughProcesses you followed to arrive at that conclusion based on the ExpertReports.
                - Try to avoid symbols and special characters in the answer.

                Provide the final report in the following JSON format only:
                {{
                  "final_report": {{
                    "query": "<>",
                    "final_answer": "< if you are confident about the answer provide the final answer else say 'Data not found in available documents.'>",
                    "citations": "<if the final_answer is found then provide the sources_used along with the page numbers>",
                    "confidence_level":"high/low/medium",
                    "reason": "<reason for the final answer provided if any>"
                  }}
                }}
            """
            finalAnswer = run_agent(final_prompt, label="Final Answer")
            log_safe("finalAnswer Output:", finalAnswer)

        # -------- Prepare final user response --------
        assistant_prompt = f"""
            You are a helpful assistant.

            Here is the user query: {json.dumps(query, ensure_ascii=False)}
            Here is the data to prepare the answer:
            {json.dumps(finalAnswer.get('final_report', ''), ensure_ascii=False)}

            Provide a detailed response to the query in clear and concise language. Use only the data provided do not add/ infer any data.
            Make sure no point provided above is missed in the final response.
            - make sure to add the detailed answer
            - make sure to add the detailed citations
            - make sure to add the confidence level
            - Try to avoid symbols and special characters in the answer.
            - Response to the user query in table format.
            - In the table just add the numeric values in the table, and rest can be in the text format.
            - in the end of the asnwer , add the sources used along with the page numbers in a separate table.
            - also add "This is an AI generated response based on the available documents. You may verify the response against actual document if needed." at the end of the asnwer.
            - Make sure your tables are in the markdown format.

            Give the response in json format:
            {{
              "response": "<detailed response in markdown language along with the table and the citations and detailed calculations if any. >"
            }}
        """
        answer = run_agent(assistant_prompt, replace_currency=True, label="User Answer")
        log_safe("Answer Output to the user:", answer)

        user_response = answer.get('response') or "I've compiled the findings, but the response field was empty."
        logging.info(f"finbot : Final answer to the user {user_response}")
        return user_response, logs

    except Exception as e:
        log_safe(f"Error in askQuery_UAT: {str(e)}")
        logging.error(f"finbot : Error in askQuery_UAT: {str(e)}")
        return "Just got hit by a error!! can you please retry asking the question!!", logs


def finbotQuery_UAT(convoID, employee_email, query):
    finbotQueryLogs=[]
    try:
        logging.info("finbot : finbotQuery function started")
        finbotQueryLogs.append("finbotQuery function started")
        finbotQueryLogs.append(f"convoID: {convoID}, employee_email: {employee_email}, query: {query}")

        classify=call_ai(
            prompt=f"""
                You are a helpful assistant that classifies user queries into 'query' or 'presentation' based on the following criteria:

                query:
                - Direct questions seeking specific financial data or explanations.

                presentation:
                - requires presentation of a specific company.
                - the user is asking for a presentation or a report format.
                - the word 'presenation' is mentioned in the query.
                
                Classify the following query: "{query}"

                Respond with either "query" or "presentation" only.
                The reponse should be in json format as:
                {{
                    "classification": "query" | "presentation"
                }}               
            """,
            role="You are the expert in financial data and numbers."
        )
        logging.info(f"finbot : Classify output: {classify}")
        finbotQueryLogs.append(f"Classify output: {classify}")
        
        if json.loads(classify)["classification"].strip().lower() == 'presentation':
            logging.info("finbot : finbot : Presentation type query identified.")
            finbotQueryLogs.append("Presentation type query identified.")
            answer=presentation(query)
            send_message(
                convo_id=convoID, 
                email=employee_email,
                query=query,
                response =answer,
                graph_generated = '',
                image_base64=''
            )
            return {"response":answer}
        else:
            logging.info(f"finbot : convoID: {convoID}, employee_email: {employee_email}, query: {query}")
            datetime_ =datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d/%m/%Y %H:%M")
            update_or_add_field(convo_id=convoID, update_data=
                        {"employee_email":employee_email,
                        "query": query, 
                        "start_dateTime":datetime_,
                        "satisfaction":None,
                        })
            
            answer, logs = askQuery_UAT(convoID, employee_email, query)
            
            finbotQueryLogs = finbotQueryLogs + logs
            
            graphAgent = Agent_UAT(
                prompt=f"""
                You are an expert in creating financial graphs. Based on the expert's answer and accompanying table below (if any graph can be formed from the tables, it should contain numbers for generating a graph), generate a Python code snippet that calls the provided function:

                    Answer: {answer}

                    Function signature:
                    def generate_financial_graph(data, labels, title, x_label, y_label, graph_type='bar')

                    Parameters:
                    data       list of numeric values  
                    labels     list of labels matching each data point  
                    title      chart title (string)  
                    x_label    x-axis label (string)  
                    y_label    y-axis label (string)  
                    graph_type one of 'bar', 'line', or 'pie'  

                    
                    Important Notes: 
                    - Please make sure to choose the data clearly so that the graph should help the finance team 
                    Return only the code snippet in this exact form:

                    result = generate_financial_graph(
                        data=...,
                        labels=...,
                        title='...',
                        x_label='...',
                        y_label='...',
                        graph_type='...'
                    )

                    If no table data can be formed, output exactly:

                    "no tables can be formed".

                    The output should be in json format
                    {{
                        "code_snippet": "..."
                    }}

                    """
            )
            
            finbotQueryLogs.append(f"Graph agent output: {graphAgent}")
            logging.info(f"finbot : Graph agent output: {graphAgent}")
            try:
                if "no tables can be formed" not in str(graphAgent).lower():
                    code_snippet = json.loads(graphAgent)['code_snippet']
                    extractedCode = extract_python_code(code_snippet)[0]
                    local_vars={}
                    exec(extractedCode, globals(), local_vars)
                    img_base64 = local_vars['result']
                    graph=True

                else:
                    img_base64 = 'no tables can be formed'
                    graph=False  
            except Exception as e:
                logging.error(f"finbot : Error in graph generation: {str(e)}")
                img_base64 = ''
                graph=False

            datetime_ =datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%d/%m/%Y %H:%M")

            update_or_add_field(convo_id=convoID, update_data=
                        {"employee_email":employee_email,
                        "query": query, 
                        "response":answer, 
                        "graph_generated":graph, 
                        "end_dateTime":datetime_,
                        "satisfaction":None,
                        "image_base64":img_base64, 
                        "logs":finbotQueryLogs
                        })
            
            send_message(
                convo_id=convoID, 
                email=employee_email,
                query=query,
                response =answer,
                graph_generated = graph,
                image_base64=img_base64
            )
            logging.info(f"finbot : Final answer sent to user: {answer}")
            # return {"status":"Sucesses","answer":answer, "img_base64":img_base64}
            return {"response":answer}
        
    except Exception as e:
        logging.error(f"finbot : Error in finbotQuery: {str(e)}")
        send_message(
            convo_id=convoID, 
            email=employee_email,
            query=query,
            response =e,
            graph_generated = '',
            image_base64=''
        )

        return {"status":"Failed","error Occured":str(e)}

def main_FinBotQuery_UAT(convoID, employee_email, query):
    # threading.Thread(target=finbotQuery_UAT, args=(convoID, employee_email, query)).start()
    r=finbotQuery_UAT(convoID, employee_email, query)
    return {"status": "Sucesses", "answer": "Your request is being processing.", "response":r}

def updateMongo(dataframe, collection_name, index_name, dbName="AIML_DB_UAT"):
    """
    Update MongoDB with the filtered DataFrame.
    """
    try:
        # dataframe_json = json.loads(dataframe)
        logging.info(dataframe)
        mongo_client = pymongo.MongoClient("mongodb+srv://AIML_DB_UAT_USR:2xiQ8K5cpfmTpCS8@poonawallacluster0.to3tz.mongodb.net/", connect=False)
        db = mongo_client[f"{dbName}"]
        collection = db[f"{collection_name}"]
        try:
            #check for each index in the existing indexes and if the key is same as index_name then save the indexValue should be the name of that index else create a new index and save the indexValue as the name of that index
            existing_indexes = collection.index_information()
            logging.info(f"finbot : Existing indexes: {existing_indexes}")
            index_exists = False
            for idx, index_info in existing_indexes.items():
                if idx == index_name and index_info['key'] == [(index_name, 1)]:
                    logging.info(f"finbot : Index '{index_name}' with same key already exists.")
                    index_exists = True
                    break
            if not index_exists:
                collection.create_index([(index_name, 1)], name=index_name, unique=True)
                logging.info(f"finbot : Index '{index_name}' created successfully.")
        except Exception as e:
            logging.error(f"Error creating index: {e}")
        try:
            # data = json.loads(dataframe_json)
            operations = []
            for record in dataframe:
                logging.info(record)
                logging.info(record[index_name])
                operations.append(
                    UpdateOne(
                        {index_name: record[f"{index_name}"]},
                        {'$set': record},
                        upsert=True
                    )
                )
            if operations:
                collection.bulk_write(operations)
                logging.info("finbot : Bulk write successful.")
        except Exception as e:
            logging.error(f"Error updating MongoDB: {e}")
        return "MongoUpdatedSucessfully"
        
    except Exception as e:
        return f"Error updating MongoDB: {e}"
    
def appendMongo(index_name, index_value,columnName, value, collectionName, db="AIML_DB_UAT"):
    client = pymongo.MongoClient("mongodb+srv://AIML_DB_UAT_USR:2xiQ8K5cpfmTpCS8@poonawallacluster0.to3tz.mongodb.net/", connect=False)
    db = db
    collection = client[db][f"{collectionName}"] 
    try:
        result = collection.update_one(
            {f"{index_name}": f"{index_value}"},
            {"$push": {f"{columnName}": value}}
        )
        if result.modified_count > 0:
            logging.info("finbot : Conversation appended successfully.")
            return {"Status":"MongoUpdatedSucessfully"}, "MongoUpdatedSucessfully"
        else:
            logging.info("finbot : No document was updated. Check if the ID exists.")
    except Exception as e:
        logging.error(f"Error appending conversation: {e}")
        return {"Status":f"Error Occured {e}"}

def call_ai(prompt, role, model="gpt-5-mini"):
    endpoint = ""
    model_name = "gpt-5-mini"
    key = ""

    # Initialize Azure OpenAI Service client with key-based authentication    
    client = AzureOpenAI(  
        azure_endpoint=endpoint,  
        api_key=key,  
        api_version="2024-12-01-preview"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            { "role": "system", "content": f"{role}" },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": f"{prompt}" 
                },
            ] } 
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content.strip().replace("₹", "INR ")

def presentation(query):
    logging.info("finbot : presentation function started")
    from pymongo import MongoClient
    r = call_ai(
        prompt=query,
        role="You are a financial analyst. Based on the query. provide the company name and the financial period in JSON format with keys 'company' , 'financial_period', 'standalone/consolidated'. If the information is not available, respond with 'Information not available'. the company name has to be in full form. like Aditya Birla Capital Limited, Bajaj Finance Limited and the financial_period has to be in format like FY25-26 Q4. output format has to be JSON only. {{ 'company': 'company name', 'financial_period': 'financial period', 'standalone/consolidated': 'standalone or consolidated' }}."
    )
    r=json.loads(r)
    logging.info(f"finbot : Extracted information: {r}")
    # Connect to MongoDB
    client = MongoClient("mongodb+srv://AIML_DB_UAT_USR:2xiQ8K5cpfmTpCS8@poonawallacluster0.to3tz.mongodb.net/")  # Replace with your MongoDB connection string

    # Select the database and collection
    db = client["AIML_DB_UAT"]  # Replace with your database name
    collection = db["quaterDataXBRL-finbot"]  # Replace with your collection name
    
    query = {
        "COMPANY NAME": {"$regex": r['company'], "$options": "i"},  # Case-insensitive match
        "FINANCIAL PERIOD": r['financial_period'],
    } 
    xbrldata_from_mongo = list(collection.find(query))
    logging.info(f"finbot : XBRL data from MongoDB: {len(xbrldata_from_mongo)}")
    xbrldata_from_mongo_update=[item for item in xbrldata_from_mongo if item['CONSOLIDATED / STANDALONE'].lower() == r['standalone/consolidated'].lower()]
    if len(xbrldata_from_mongo_update)>0:
        xbrldata_from_mongo=xbrldata_from_mongo_update

    answer=call_ai(
        prompt=f"The user is asking for company details : {r}. \n data:{xbrldata_from_mongo}",
        role="""You are a financial analyst, Based on the data extract the following information in json format: 
        - AUM
        - Interest Income
        - Interest Expense, 
        - PBT (calculation = PPOP - Provision)
        - PAT
        - NIM  (calculation = Interest Inc- Int Exp)
        - NIM (including other income) (calculation = NIM + Other Income)
        - Operating Expenses (expenses excluding Finance cost and Impairement)
        - PPOP (calculation = NIM(Incl other Inc) - Opex)
        - Provisions
        - Exceptional items (net of tax) 
        - PAT with exceptional items
        
        - Yield % (if directtly not found use this calculation = Int Income / Avg Asset under Finance (AUF) * 4)
        - COF % (if directtly not found use this calculation = Int Exp / Avg Asset under Finance (AUF) * 4)
        - NIM % (if directtly not found use this calculation = NIM/ Avg  Asset under Finance (AUF) * 4)
        - Other Income to ATA (if directtly not found use this calculation = Other Income / Avg Asset under Finance (AUF) * 4)
        - Total Income % (if directtly not found use this calculation = NIMincl. Other Inc / Avg Asset under Finance (AUF) * 4)
        - Opex to ATA (if directtly not found use this calculation =  (Other Income / Avg Asset under Finance (AUF)) * 4)
        - PPOP %  (if directtly not found use this calculation =  PPOP/Avg Asset under Finance (AUF) * 4)
        - Credit Cost %  (if directtly not found use this calculation =  Credit Cost / Avg Asset under Finance (AUF) * 4)
        - ROA (%) (after fetching the value multiply by 4 for quaterly data)
        - ROA (%) (with exceptional item)
        - ROE (%)
        - Opex to NII (%)  (if directtly not found use this calculation =  Opex/ NIM incl Other Income)
        - GNPA%
        - NNPA%
        - PCR %
        - Net Worth
        - Total Borrowings
        - Avg AUF  (calculation =  PAT/ROA * 4 quarterly , otherwise PAT/ROA for annual)
        - No. of shares(in Lakhs)
        - Book Value


        Note:
        - just give the metric name
        - fetch data from quater data only
        - some calculation are mentioned above, use them wherever applicable
        - if the calulcation is not provided, then try yourself based on the data provided
        - in the data some values are in crores, some in lakhs, some in thousands, make sure to convert them all to the same unit (preferably crores) before providing the final value.
        - in the output just provide the metric name.
        - Provide the data in the json format. 
        - Don't assume any value, extract if it is cleary stated or can be calculated from the data. like for example, dont assume AUM as asset under finance or AUM is taken as the loan book.  
        - also if you can caluclte then call out the calculation explicitly in the json response.
        
        The repondr should be in the format:
        {{
            "metric":[{
                "name":"Total Income",
                "value":"value if it can be found or calculated from the data else #NA",
                "calculation":"if any calculation done"
                "source":"from where the value is taken with the filename along with the page number"
            },
            {
                "name":"AUM",
                "value":"value if it can be found or calculated from the data else #NA",
                "calculation":"if any calculation done"
                "source":"from where the value is taken with the filename along with the page number"
            },]
        }}
        """
    )
    answer=json.loads(answer)
    logging.info(f"finbot : Presentation answer: {answer}")
    for i in answer['metric']:
        if i['value'] == "#NA":
            print(i)
            logging.info(f"finbot : blank data from updated MongoDB for metric: {i['name']}")

    from pymongo import MongoClient

    # Connect to MongoDB
    client = MongoClient("mongodb+srv://AIML_DB_UAT_USR:2xiQ8K5cpfmTpCS8@poonawallacluster0.to3tz.mongodb.net/")  # Replace with your MongoDB connection string

    # Select the database and collection
    db = client["AIML_DB_UAT"]  # Replace with your database name
    collection = db["financialData-finbot"]  # Replace with your collection name

    # Fetch data from the collection
    # query = {} 
    query = {
        "company_info.Company Name": {"$regex": r['company'], "$options": "i"},  # Case-insensitive match
        "company_info.FINANCIAL PERIOD": r['financial_period']
    } # Define your query here, e.g., {"key": "value"} for filtering
    data_from_mongo = list(collection.find(query))
    logging.info(f"finbot : Financial data from MongoDB: {len(data_from_mongo)}")

    if len(data_from_mongo)>0:
        print("Merging financial data from updated MongoDB records.")
        merged_financial_data = {}
        for doc in data_from_mongo:
            if 'financial_metrics' in doc and isinstance(doc['financial_metrics'], dict):
                merged_financial_data.update(doc['financial_metrics'])
        merged_financial_data


    if not data_from_mongo:
        logging.info("finbot : No data found in MongoDB for the given query.")
    else:
        answer=call_ai(
            prompt=f"The user is asking for company details : {r}. \n data:{data_from_mongo} also I have some answers {answer}",
            role="""You are a financial analyst, Based on the data extract the following information in json format: 
            some values are already provided in the answer, do verify that and update if needed 
            - Asset under Management (AUM)
            - Asset under Finance (AUF) (same as avg AUF based on the data provided)
            - Interest Income
            - Interest Expense, 
            - PBT (if directtly not found use this calculation = PPOP - Provision)
            - PAT
            - NIM  ( if directtly not found use this calculation = Interest Inc- Int Exp)
            - NIM (including other income) (if directtly not found use this calculation = NIM + Other Income)
            - Operating Expenses (expenses excluding Finance cost and Impairement)
            - PPOP (if directtly not found use this calculation = NIM(Incl other Inc) - Opex)
            - Other income (if directtly not found use this calculation = Sum of all incomes except interest income or net interest income)
            - Provisions
            - Exceptional items (net of tax) 
            - PAT with exceptional items

            - Yield % (value *4 if directtly not found use this calculation = Int Income / Avg Asset under Finance (AUF) * 4)
            - COF % (value *4  if directtly not found use this calculation = Int Exp / Avg Asset under Finance (AUF) * 4)
            - NIM % (value *4  if directtly not found use this calculation = NIM/ Avg  Asset under Finance (AUF) * 4)
            - Other Income to ATA (value *4  if directtly not found use this calculation = Other Income / Avg Asset under Finance (AUF) * 4)
            - Total Income % (value *4  if directtly not found use this calculation = NIMincl. Other Inc / Avg Asset under Finance (AUF) * 4)
            - Opex to ATA (value *4  if directtly not found use this calculation =  (Other Income / Avg Asset under Finance (AUF)) * 4)
            - PPOP %  (value *4  if directtly not found use this calculation =  PPOP/Avg Asset under Finance (AUF) * 4)
            - Credit Cost %  (value *4  if directtly not found use this calculation =  Credit Cost / Avg Asset under Finance (AUF) * 4)
            - ROA (%) (after fetching the value multiply by 4 for quaterly data)
            - ROA (%) (with exceptional item)
            - ROE (%)
            - Opex to NII (%)  (value *4  if directtly not found use this calculation =  Opex/ NIM incl Other Income)
            - GNPA%
            - NNPA%
            - PCR %
            - Net Worth
            - Total Borrowings
            - Avg AUF  (value *4  if directtly not found use this calculation =  PAT/ROA * 4 quarterly , otherwise PAT/ROA for annual)
            - No. of shares(in Lakhs)
            - Book Value


            Note:
            - just give the metric name
            - fetch data from quater data only
            - some calculation are mentioned above, use them wherever applicable
            - if the calulcation is not provided, then try yourself based on the data provided
            - in the data some values are in crores, some in lakhs, some in thousands, make sure to convert them all to the same unit (preferably crores) before providing the final value.
            - in the output just provide the metric name.
            - Provide the data in the json format. 
            - Don't assume any value, extract if it is cleary stated or can be calculated from the data. like for example, dont assume AUM as asset under finance or AUM is taken as the loan book.  
            - also if you can caluclte then call out the calculation explicitly in the json response.
            - at the end add this line "This is an AI generated response based on the available documents. You may verify the response against actual document if needed"
            The reponse should be in the format:
            {{
                "metric":[{
                    "name":"Total Income",
                    "value":"value if it can be found or calculated from the data else #NA",
                    "calculation":"if any calculation done",
                    "source":"from where the value is taken with the filename along with the page number"
                },
                {
                    "name":"AUM",
                    "value":"value if it can be found or calculated from the data else #NA",
                    "calculation":"if any calculation done",
                    "source":"from where the value is taken with the filename along with the page number"
                },]
                "note":"This is an AI generated response based on the available documents. You may verify the response against actual document if needed"
            }}
            """
        )
        answer=json.loads(answer)
        answer

    print('answer ', answer)

    
    for i in answer['metric']:
        if i['value'] == "#NA":
            print(i)

    return answer
