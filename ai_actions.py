# AI Contacts Search Assistant
# Copyright (C) 2023 Alex Furmansky, Magnetic Ventures LLC
#
# This file is part of the AI Contacts Search Assistant.
#
# AI Contacts Search Assistant is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI Contacts Search Assistant is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with AI Contacts Search Assistant. If not, see <https://www.gnu.org/licenses/>.

import os
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging


# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a function to reformat experiences using OpenAI's GPT-3
def reformat_experiences(experiences):
    # Initialize the OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # Create a prompt template
    system_template = (
        "Below is the employment history of a person. Write a one paragraph summary of this person based on this experience. "
        "Specifically, include the person's strengths and skill sets. Your analysis must be based only on the employment history "
        "and your factual knowledge of the companies where this person worked. Imagine you are describing this person to a recruiter "
        "so the recruiter could best determine the best roles for this person. \n\n Experience: {experiences}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_template),
        ('user', '{experiences}')
    ])
    
    # Create an output parser
    parser = StrOutputParser()
    
    # Create a Langchain LLMChain
    chain = prompt_template | llm | parser
    
    # Generate a response
    response = chain.invoke({"experiences": experiences})
    logging.info(f"LLM response reformat_experiences: {response}")
    
    return response

