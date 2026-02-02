# main.py
import asyncio
from logger import logger
import clients
from bot import process_message

async def main(test_email, test_question):
    """Main function for testing the bot."""
    try:
        # Initialize clients
        clients.init_clients()
        await clients.init_db()
        logger.info("Taxation Bot initialized successfully")
        
        print("\n" + "="*80)
        print("TAXATION BOT - TEST RUN")
        print("="*80)
        print(f"\nEmail: {test_email}")
        print(f"Question: {test_question}")
        
        # Process message
        result = await process_message(test_question, test_email)
        
        # Display results
        print("\n" + "-"*80)
        print("RETRIEVED CONTEXT:")
        print("-"*80)
        if result.get('context'):
            print(result['context'])
        else:
            print("No context retrieved")
        
        print("\n" + "-"*80)
        print("RESPONSE:")
        print("-"*80)
        print(f"Status: {result['status']}")
        print(f"Confidence Score: {result['confidence_score']}")
        print(f"\nAnswer:\n{result['answer']}")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        await clients.close_clients()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        test_email = "test4@example.com"
        test_question = "please explain me the rule related to tax deduction at source for salaried employees"
        asyncio.run(main(test_email, test_question))
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)