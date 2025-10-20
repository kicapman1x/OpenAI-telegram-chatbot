import os
import logging
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
openai_model = None
client = None 
bot_token = None

def bootstrap():
    global client, openai_model, bot_token
    load_dotenv()

    bot_token = os.environ.get("bot_token")
    logdir = os.environ.get("log_directory", ".")
    loglvl = os.environ.get("log_level", "INFO").upper()
    openai_model=os.environ.get("openai_model")

    client = OpenAI(
        api_key=os.environ.get("openai_token")
    )

    log_level = getattr(logging, loglvl, logging.INFO)

    logging.basicConfig(
        filename=f'{logdir}/telebot.log',
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info('*********** Bootstrap completed ***********')


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
        response = client.responses.create(
            model=openai_model,
            instructions="You are a friendly UTM campus assistance that helps students navigate around campus and around campus rules and regulations.",
            input=message,
        )
        gpt_reply=response.output_text
        return gpt_reply

    except Exception as e: 
        logger.error(f"Error with OpenAI API: {e}")
        return "Apologies, I could not process that request. Please contact the administrator."

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