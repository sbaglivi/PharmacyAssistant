# Pharmacy assistant
The prompt uses a standard format: we tell the agent who he is, an overview of what his role is and then a list of more specific guidelines. For each guideline, we specify what not to do and a course of action to prefer instead.  
We use few shot prompting: we provide the model with a handful of examples of short conversations to help the LLM recognize some of the scenarios defined in the guidelines and adhere to the required behaviour.  
I used markdown in the prompt to help convey the hierarchy of information given to the LLM.  
  
In my testing (mostly done with gemini 2.5 flash / flash-lite), the agent is committed to its task and seems to respect the instructions.  
I personally tried "jailbreaking" it with an emotional situation (a loved one in danger of passing) and with messages in other languages or encoded (chinese - ai translated, base64 encoded requests).  
With a first version of the prompt, I was of the opinion that the agent showed too much resistance to give information about which medications respect user given criteria and dosage informations. That lead me to introduce a couple of phrases and examples specifically targeted at slightly widening the range of tasks it was willing to undertake.  

## Limitations
Since the example targets an interaction through a simple chat interface (like ChatGPT) the hand-off is "fake": there's no human in the loop, nor other agents with a contextualized prompt for other languages.  
In a real system:
- we might use structured outputs to let the agent express his desire to involve another agent or a human in the conversation, and through a framework we would route the current conversation to the designated individual;
- we'd have to check whether a human is readily available or whether there is a wait time involved, in which case we might suggest the user to create a ticket to receive an async response ASAP;
- we'd probably monitor whether the user keeps trying to derail the conversation (through the same agent or a specialized one), and if that were to happen enough turns in a row we could just end the conversation. 

## AI usage
I used AI to create the JSON file containing the list of available drugs.  
I also used AI to automate some conversations between the pharmacy assistant and either an LLM pretending to be a regular client or a LLM that had access to the prompt of the assistant and had the specific instruction of trying to derail the agent.  
I also created a small LLM as a judge with a rubric to have an external opinion on how well the agent was adhering to the guidelines.  
Coding wise I also used AI to add a retry mechanism to API calls made with the google genai python library, since apparently it does not have one out of the box.  

## Other observations
I'm not sure if the adversarial agent was acting with a plan, but the pharmacy assistant sometimes got confused when the adversary repeated the messages from the assistant and pretended to be an assitant as well, almost in an attempt to reverse roles ([example](examples/long_automated_chat.json)). I could add an explicit phrase in the prompt or an example but since it was not breaking any guidelines and it didn't feel like a behaviour it was likely to encounter in a real world situation, I haven't done so.  
Outside of these events, I personally did not get the agent to make conversation, suggest treatments or diagnosis, or try to take the role of an healthcare provider.  
