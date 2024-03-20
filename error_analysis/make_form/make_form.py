from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools
import json

SCOPES = "https://www.googleapis.com/auth/forms.body"
DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

store = file.Storage("token.json")
creds = None
if not creds or creds.invalid:
  flow = client.flow_from_clientsecrets("credentials.json", SCOPES)
  creds = tools.run_flow(flow, store)

form_service = discovery.build(
    "forms",
    "v1",
    http=creds.authorize(Http()),
    discoveryServiceUrl=DISCOVERY_DOC,
    static_discovery=False,
)

# Request body for creating a form
NEW_FORM = {
    "info": {
        "title": "IMDb_review_human_annotation",
    }
}

# Creates the initial form
result = form_service.forms().create(body=NEW_FORM).execute()

with open("../errors_to_analyze.json") as f:
    reviews = json.loads(f.read())

for idx,rewiew in enumerate(reviews):
    NEW_QUESTION = {
        "requests": [
            {
                "createItem": {
                    "item": {
                        "title": (
                           f'{str(idx).zfill(3)}: {rewiew["txt"].replace("<br /><br />","")}'
                        ),
                        "questionItem": {
                            "question": {
                                "required": False,
                                "choiceQuestion": {
                                    "type": "RADIO",
                                    "options": [
                                        {"value": "positive"},
                                        {"value": "negative"},
                                        {"value": "mixed/neutral"}
                                    ],
                                    "shuffle": False,
                                },
                            }
                        },
                    },
                    "location": {"index": 0},
                }
            }
        ]
    }
    question_setting = (
        form_service.forms()
        .batchUpdate(formId=result["formId"], body=NEW_QUESTION)
        .execute()
    )

# Prints the result to show the question has been added
get_result = form_service.forms().get(formId=result["formId"]).execute()