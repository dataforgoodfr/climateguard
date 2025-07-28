climatesafeguards_template = """
{# ───── header (system message) ───── #}
{{- "<|im_start|>system\n" -}}
  {{- "## Metadata\n\n" -}}
  {{- "Role: Climate Expert\n" -}}
  {{- "Objective: Detecting Climate Related Disinformation" -}}
  {{- "You are an assistant helping editors to moderate TV and radio content. You will be provided with a prompt containing transcribed text from a tv or radio program. Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places,cor may include some phonetic transcription. Even if the text is not in english, analyze it seemlessly.\n\nTask: Determine if the text promotes climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it. \n\nInstructions:\n1) Your verdict should be true or false.\n2) Format your verdict always using XML tags like this: <misinformation>verdict</misinformation>" -}}
  {{- "\n\n" -}}
{{- "<|im_end|>\n" -}}

{# ───── main loop ───── #}
{%- for message in messages -%}
    {%- set content = message.content if message.content is string else "" -%}
    {%- if message.role == "user" -%}
        {{ "<|im_start|>" + "user\n"  + content + "<|im_end|>\n" }}
    {%- elif message.role == "assistant" -%}
        {% generation %}
        {{ "<|im_start|>assistant\n" + content.lstrip("\n") + "<|im_end|>\n" }}
        {% endgeneration %}
    {%- endif -%}
{%- endfor -%}
{# ───── generation prompt ───── #}
{%- if add_generation_prompt -%}
    {{ "<|im_start|>assistant\n" }}
{%- endif -%}
"""

climatesafeguards_template_chatml = """
{# ───── header (system message) ───── #}
{{- "<|startoftext|><|im_start|>system\n" -}}
  {{- "## Metadata\n\n" -}}
  {{- "Role: Climate Expert\n" -}}
  {{- "Objective: Detecting Climate Related Disinformation" -}}
  {{- "You are an assistant helping editors to moderate TV and radio content. You will be provided with a prompt containing transcribed text from a tv or radio program. Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places,cor may include some phonetic transcription. Even if the text is not in english, analyze it seemlessly.\n\nTask: Determine if the text promotes climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it. \n\nInstructions:\n1) Your verdict should be true or false.\n2) Format your verdict always using XML tags like this: <misinformation>verdict</misinformation>" -}}
  {{- "\n\n" -}}
{{- "<|im_end|>\n" -}}

{# ───── main loop ───── #}
{%- for message in messages -%}
    {%- set content = message.content if message.content is string else "" -%}
    {%- if message.role == "user" -%}
        {{ "<|im_start|>" + "user\n"  + content + "<|im_end|>\n" }}
    {%- elif message.role == "assistant" -%}
        {% generation %}
        {{ "<|im_start|>assistant\n" + content.lstrip("\n") + "<|im_end|>\n" }}
        {% endgeneration %}
    {%- endif -%}
{%- endfor -%}
{# ───── generation prompt ───── #}
{%- if add_generation_prompt -%}
    {{ "<|im_start|>assistant\n" }}
{%- endif -%}
"""