import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import logging
import json
import numpy as np
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
# from asyncio import to_thread

logger = logging.getLogger(__name__)
openai_model = None
client = None 
bot_token = None
chroma_client = None
collection = None
openai_ef = None 

def bootstrap():
    #ENV VAR
    global client, openai_model, bot_token, chroma_client, collection, openai_ef
    load_dotenv()

    bot_token = os.environ.get("bot_token")
    logdir = os.environ.get("log_directory", ".")
    loglvl = os.environ.get("log_level", "INFO").upper()
    openai_model=os.environ.get("openai_model")
    embed_model=os.environ.get("embed_model")
    chromadir = os.environ.get("chroma_directory", ".")
    fill_db = os.environ.get("fill_db", "False").lower() == "true"
    openai_api_key=os.environ.get("openai_token")
    #OpenAI client
    client = OpenAI(
        api_key=openai_api_key
    )

    #OpenAI ef
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=embed_model
    )

    #logger
    log_level = getattr(logging, loglvl, logging.INFO)

    logging.basicConfig(
        filename=f'{logdir}/telebot.log',
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    #ChromaDB 
    chroma_client = chromadb.PersistentClient(path=chromadir)
    collection = chroma_client.get_or_create_collection(name="utm_jb_campus_locations")
    #update db
    if (fill_db):
        logger.info('Collection has documents')
        writeChromaDB()
        logger.info('Database has been populated with initial data.')
    else:
        logger.info('Database already populated or fill_db is False. Skipping population.')
    logger.info('*********** Bootstrap completed ***********')

def writeChromaDB():
    with open("chroma_input.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f'Provisioning ids')
    ids = [location['id'] for location in data]
    logger.info(f'Provisioning docs')
    documents = [location['name'] for location in data]
    logger.info(f'Provisioning metadatas')
    metadatas = [{
        "id": location['id'],
        "name": location['name'],
        "address": location['address'],
        "latitude": location['latitude'],
        "longitude": location['longitude']
    } for location in data]
    logger.info('Provisioning embeddings')
    logger.info(documents)
    embedding=openai_ef(documents)
    logger.info(embedding)
    # embeddings = [openai_ef(location['name'])[0].tolist() for location in data]
    # logger.info(f"Generated embeddings for data, example: {embeddings[:1]}")
    logger.info(f'Updating collections')
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embedding
    )
    logger.info('*********** Data written to ChromaDB ***********')

async def start(update, context):
    await update.message.reply_text("Hello! I'm your friendly UTM Campus Assistant. How may I assist you today?")

async def healthCheck(update, context):
    await update.message.reply_text("Application is healthy! Bot is running fine!")

async def handleMessage(update, context):
    message = update.message.text 
    logger.info(f'Message received: {message}')
    replytext = 'Message not processed'
    #In case you need to deal with messages not being sent to OpenAI as well
    if message.startswith('##'):
        logger.info('Escape triggered!')
        replytext=await handleEscape(message)
    else:
        replytext=await handleGPT(message)
    await update.message.reply_text(replytext)

async def handleEscape(message):
    messageReply='Escape has been triggered - message diverted from OpenAI'
    return messageReply

async def handleGPT(message):
    try:
        chromadb_data_result=None 
        result_docs=''
        result_metas=''
        #preferrably a validator, for testing, we will just leave it as if True
        if (True):
            chromadb_data_result=await query_chroma_db(message)
        if chromadb_data_result: 
            logger.info("documents returned from ChromaDB")
            result_docs=str(chromadb_data_result['documents'])
            result_metas=str(chromadb_data_result['metadatas'])
        else:
            logger.error("No valid documents returned from ChromaDB")
        response = client.responses.create(
            model=openai_model,
            instructions="""
            You are a friendly UTM campus assistance that helps students navigate around campus and around campus rules and regulations. Upstream from asking you, I have ran a chromadb query and I will provide the relevant data below, you should try your best as possible to use the data given, if suitable. 
            
            The data: 

            """+result_docs+"""
            """+result_metas+"""
            """,
            input=message,
        )
        gpt_reply=response.output_text
        return gpt_reply

    except Exception as e: 
        logger.error(f"Error with OpenAI API: {e}")
        return "Apologies, I could not process that request. Please contact the administrator."

async def query_chroma_db(message):
    query_embeddings = openai_ef([message])
    # query_embeddings = query_embeddings[0].tolist()
    logger.info(f"Received message and embedded to: {query_embeddings}")
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=3
    )
    logger.info(f"Returning: {results}")
    return results

def main():
    logger.info('***********Starting UTM Campus Assistance Bot***********')

    bootstrap()

    logger.info(f'Bot token has been set as : {bot_token}')
    logger.info(f'Model has been set as: {openai_model}')

    if not bot_token:
        logger.error("Bot token is not found! Exiting!")
        return
    
    application = Application.builder().token(bot_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("health", healthCheck))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handleMessage))

    application.run_polling()

if __name__ == '__main__':
    main()