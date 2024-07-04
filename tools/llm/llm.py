from openai import OpenAI

prompt_dict = {
    "kimi": [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in both Chinese and English conversations.",
        },
        {
            "role": "user",
            "content": "Hello, please note that the text you generate should follow a natural conversational tone. Your responses will be converted to speech by a TTS model, and please keep your answers within 100 characters. Also, use only commas and periods for punctuation, and convert numbers to words.",
        },
        {
            "role": "assistant",
            "content": "Got it, I will generate text in a natural conversational tone, keep my responses within 100 characters, use only commas and periods for punctuation, and convert numbers to words. Let's start the conversation.",
        },
    ],
    "deepseek": [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "Hello, please note that the text you generate should follow a natural conversational tone. Your responses will be converted to speech by a TTS model, and please keep your answers within 100 characters. Also, use only commas and periods for punctuation, and convert numbers to words.",
        },
        {
            "role": "assistant",
            "content": "Got it, I will generate text in a natural conversational tone, keep my responses within 100 characters, use only commas and periods for punctuation, and convert numbers to words. Let's start the conversation.",
        },
    ],
    "deepseek_TN": [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": "Hello, we are now processing text input for TTS. Please input a text, and convert any numbers to words, using only commas and periods for punctuation.",
        },
        {
            "role": "assistant",
            "content": "Got it, I am now processing the text input for TTS. This is generally called text normalization. Please input the text.",
        },
        {"role": "user", "content": "We paid $123 for this desk."},
        {
            "role": "assistant",
            "content": "We paid one hundred and twenty three dollars for this desk.",
        },
        {"role": "user", "content": "For more details, please call 010-724654"},
        {"role": "assistant", "content": "For more details, please call zero one zero, seven two four six five four"},
        {"role": "user", "content": "Lawson announced it will delist on July 24, with over 6000 stores in China!"},
        {
            "role": "assistant",
            "content": "Lawson announced it will delist on July twenty-fourth, with over six thousand stores in China.",
        },
    ],
}


class ChatOpenAI:
    def __init__(self, api_key, base_url, model):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def call(self, user_question, temperature=0.3, prompt_version="kimi", **kwargs):

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt_dict[prompt_version]
            + [
                {"role": "user", "content": user_question},
            ],
            temperature=temperature,
            **kwargs
        )
        return completion.choices[0].message.content
