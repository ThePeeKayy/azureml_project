"""
Azure Function App for ADF Pipeline Integration
Provides REST API to fetch flight data and trigger pipelines
"""

import azure.functions as func
import logging
import json
import os
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.identity import DefaultAzureCredential
import requests

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Configuration
STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_PIPELINE_CONNECTION_STRING")
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP")
FACTORY_NAME = "adf-azureproj1-portfolio"
PIPELINE_NAME = "pl_flight_data_ingestion"

def get_adf_client():
    """Initialize ADF client with managed identity"""
    try:
        credential = DefaultAzureCredential()
        client = DataFactoryManagementClient(credential, SUBSCRIPTION_ID)
        return client
    except Exception as e:
        logging.error(f"Failed to initialize ADF client: {str(e)}")
        return None

@app.route(route="flights", methods=["GET"])
def get_flights(req: func.HttpRequest) -> func.HttpResponse:
    """
    Fetch latest flight data *from blob*, parse OpenSky JSON,
    and return structured flight objects instead of blob URLs.
    """
    logging.info('Fetching latest flight data INCLUDING blob content')

    try:
        if not STORAGE_CONNECTION_STRING:
            raise ValueError("Storage connection string not configured")

        blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client("raw-flights")

        blobs = list(container_client.list_blobs())
        if not blobs:
            return func.HttpResponse(
                json.dumps({
                    "status": "empty",
                    "message": "No flight data found.",
                    "flights": []
                }),
                mimetype="application/json",
                status_code=200
            )

        blobs.sort(key=lambda x: x.last_modified, reverse=True)
        latest_blob = blobs[0]

        blob_client = container_client.get_blob_client(latest_blob.name)
        blob_data = blob_client.download_blob().readall().decode("utf-8")

        json_data = json.loads(blob_data)
        raw_states = json_data.get("states", [])

        flights = [
            {
                "icao24": s[0],
                "callsign": s[1].strip() if s[1] else None,
                "origin_country": s[2],
                "longitude": s[5],
                "latitude": s[6],
                "baro_altitude": s[7],
                "on_ground": s[8],
                "velocity": s[9],
                "vertical_rate": s[11],
                "last_contact": s[4],
            }
            for s in raw_states[:50]
        ]

        # Construct response
        response = {
            "status": "success",
            "latest_update": latest_blob.last_modified.isoformat(),
            "file_name": latest_blob.name,
            "flight_count": len(flights),
            "flights": flights,
        }

        return func.HttpResponse(
            json.dumps(response),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error fetching full flight data: {str(e)}")

        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "error": str(e),
                "flights": []
            }),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="triggers", methods=["GET"])
def get_triggers(req: func.HttpRequest) -> func.HttpResponse:
    """Get status of all pipeline triggers from ADF"""
    logging.info('Getting trigger status from Data Factory')
    
    try:
        # Check if required config is available
        if not all([SUBSCRIPTION_ID, RESOURCE_GROUP]):
            raise ValueError("Missing ADF configuration")
        
        # Get ADF client
        client = get_adf_client()
        if not client:
            raise ValueError("Failed to create ADF client")
        
        # Query real triggers from ADF
        triggers_list = []
        triggers = client.triggers.list_by_factory(RESOURCE_GROUP, FACTORY_NAME)
        
        for trigger in triggers:
            trigger_run = None
            try:
                # Try to get the latest run
                runs = client.trigger_runs.query_by_factory(
                    RESOURCE_GROUP,
                    FACTORY_NAME,
                    {
                        "lastUpdatedAfter": (datetime.now() - timedelta(hours=24)).isoformat() + "Z",
                        "lastUpdatedBefore": datetime.now().isoformat() + "Z",
                        "filters": [
                            {
                                "operand": "TriggerName",
                                "operator": "Equals",
                                "values": [trigger.name]
                            }
                        ]
                    }
                )
                if runs.value and len(runs.value) > 0:
                    trigger_run = runs.value[0]
            except Exception as e:
                logging.warning(f"Could not fetch trigger runs: {str(e)}")
            
            # Determine trigger type
            trigger_type = "schedule"
            if hasattr(trigger.properties, 'type_properties'):
                if "TumblingWindow" in str(type(trigger.properties)):
                    trigger_type = "tumbling-window"
                elif "BlobEvents" in str(type(trigger.properties)):
                    trigger_type = "storage-event"
            
            trigger_data = {
                "id": trigger.name,
                "name": trigger.name,
                "type": trigger_type,
                "status": "Active" if trigger.properties.runtime_state == "Started" else "Inactive",
                "description": f"{trigger.type} trigger for {PIPELINE_NAME}",
                "lastRun": trigger_run.trigger_run_timestamp.isoformat() if trigger_run else None,
                "nextRun": "Checking..." if trigger.properties.runtime_state == "Started" else "Stopped"
            }
            triggers_list.append(trigger_data)
        
        return func.HttpResponse(
            json.dumps({"triggers": triggers_list, "source": "real"}),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error fetching triggers: {str(e)}")
        
        # Fallback to demo data
        triggers = [
            {
                "id": "1",
                "name": "Schedule Trigger (Every 10 minutes)",
                "type": "schedule",
                "status": "Active",
                "description": "Runs pipeline every 10 minutes to fetch latest flight data",
                "recurrence": "Every 10 Minutes",
                "lastRun": (datetime.now() - timedelta(minutes=10)).isoformat(),
                "nextRun": (datetime.now() + timedelta(minutes=10)).isoformat()
            },
            {
                "id": "2",
                "name": "Tumbling Window Trigger (Hourly)",
                "type": "tumbling-window",
                "status": "Active",
                "description": "Aggregates data every hour with retry capability",
                "recurrence": "Every 1 Hour",
                "lastRun": (datetime.now() - timedelta(hours=1)).isoformat(),
                "nextRun": (datetime.now() + timedelta(hours=1)).isoformat(),
                "maxConcurrency": 1,
                "retryCount": 3
            },
            {
                "id": "3",
                "name": "Storage Event Trigger",
                "type": "storage-event",
                "status": "Active",
                "description": "Triggers on new file upload to raw-flights container",
                "event": "Blob Created",
                "container": "raw-flights",
                "lastRun": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "nextRun": "Event-driven"
            }
        ]
        
        return func.HttpResponse(
            json.dumps({"triggers": triggers, "source": "demo", "error": str(e)}),
            mimetype="application/json",
            status_code=200
        )


@app.route(route="pipeline-status", methods=["GET"])
def get_pipeline_status(req: func.HttpRequest) -> func.HttpResponse:
    """Get recent pipeline run history from ADF"""
    logging.info('Getting pipeline run history')
    
    try:
        # Check if required config is available
        if not all([SUBSCRIPTION_ID, RESOURCE_GROUP]):
            raise ValueError("Missing ADF configuration")
        
        # Get ADF client
        client = get_adf_client()
        if not client:
            raise ValueError("Failed to create ADF client")
        
        # Query pipeline runs from last 24 hours
        filter_params = {
            "lastUpdatedAfter": (datetime.now() - timedelta(hours=24)).isoformat() + "Z",
            "lastUpdatedBefore": datetime.now().isoformat() + "Z",
            "filters": [
                {
                    "operand": "PipelineName",
                    "operator": "Equals",
                    "values": [PIPELINE_NAME]
                }
            ]
        }
        
        pipeline_runs = client.pipeline_runs.query_by_factory(
            RESOURCE_GROUP,
            FACTORY_NAME,
            filter_params
        )
        
        runs = []
        for run in sorted(pipeline_runs.value, key=lambda x: x.run_start, reverse=True)[:10]:  # Get last 10 runs
            duration = None
            if run.run_end and run.run_start:
                duration = int((run.run_end - run.run_start).total_seconds())
            
            runs.append({
                "runId": run.run_id,
                "status": run.status,
                "startTime": run.run_start.isoformat() if run.run_start else None,
                "endTime": run.run_end.isoformat() if run.run_end else None,
                "duration": duration,
                "pipeline": run.pipeline_name,
                "triggeredBy": run.invoked_by.name if hasattr(run, 'invoked_by') else "Manual"
            })
        
        return func.HttpResponse(
            json.dumps({"runs": runs, "source": "real"}),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error fetching pipeline status: {str(e)}")
        
        # Fallback to demo data
        runs = [
            {
                "runId": f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "status": "Succeeded",
                "startTime": (datetime.now() - timedelta(minutes=10)).isoformat(),
                "endTime": (datetime.now() - timedelta(minutes=8)).isoformat(),
                "duration": 120,
                "pipeline": PIPELINE_NAME,
                "triggeredBy": "Schedule Trigger"
            },
            {
                "runId": f"run-{(datetime.now() - timedelta(minutes=20)).strftime('%Y%m%d-%H%M%S')}",
                "status": "Succeeded",
                "startTime": (datetime.now() - timedelta(minutes=20)).isoformat(),
                "endTime": (datetime.now() - timedelta(minutes=18)).isoformat(),
                "duration": 115,
                "pipeline": PIPELINE_NAME,
                "triggeredBy": "Schedule Trigger"
            },
            {
                "runId": f"run-{(datetime.now() - timedelta(minutes=30)).strftime('%Y%m%d-%H%M%S')}",
                "status": "Succeeded",
                "startTime": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "endTime": (datetime.now() - timedelta(minutes=28)).isoformat(),
                "duration": 118,
                "pipeline": PIPELINE_NAME,
                "triggeredBy": "Schedule Trigger"
            }
        ]
        
        return func.HttpResponse(
            json.dumps({"runs": runs, "source": "demo", "error": str(e)}),
            mimetype="application/json",
            status_code=200
        )


@app.route(route="trigger-pipeline", methods=["POST"])
def trigger_pipeline(req: func.HttpRequest) -> func.HttpResponse:
    """Manually trigger the ADF pipeline"""
    logging.info('Manually triggering pipeline')
    
    try:
        # Check if required config is available
        if not all([SUBSCRIPTION_ID, RESOURCE_GROUP]):
            raise ValueError("Missing ADF configuration")
        
        # Get ADF client
        client = get_adf_client()
        if not client:
            raise ValueError("Failed to create ADF client")
        
        # Trigger the pipeline
        run_response = client.pipelines.create_run(
            RESOURCE_GROUP,
            FACTORY_NAME,
            PIPELINE_NAME
        )
        
        response = {
            "status": "triggered",
            "runId": run_response.run_id,
            "message": "Pipeline triggered successfully",
            "pipeline": PIPELINE_NAME,
            "timestamp": datetime.now().isoformat(),
            "source": "real"
        }
        
        return func.HttpResponse(
            json.dumps(response),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error triggering pipeline: {str(e)}")
        
        # Return simulated response
        run_id = f"manual-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        response = {
            "status": "simulated",
            "runId": run_id,
            "message": "Pipeline trigger simulated (using demo mode)",
            "pipeline": PIPELINE_NAME,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "source": "demo"
        }
        
        return func.HttpResponse(
            json.dumps(response),
            mimetype="application/json",
            status_code=200
        )


@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    
    # Check configuration status
    config_status = {
        "storage_configured": bool(STORAGE_CONNECTION_STRING),
        "subscription_configured": bool(SUBSCRIPTION_ID),
        "resource_group_configured": bool(RESOURCE_GROUP)
    }
    
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "configuration": config_status
        }),
        mimetype="application/json",
        status_code=200
    )


@app.route(route="opensky-direct", methods=["GET"])
def get_opensky_direct(req: func.HttpRequest) -> func.HttpResponse:
    """Fetch data directly from OpenSky API (for testing)"""
    logging.info('Fetching data directly from OpenSky API')
    
    try:
        response = requests.get(
            "https://opensky-network.org/api/states/all",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Return summary
            summary = {
                "timestamp": data.get("time"),
                "total_flights": len(data.get("states", [])),
                "sample_flight": data.get("states", [[]])[0] if data.get("states") else None,
                "status": "success"
            }
            
            return func.HttpResponse(
                json.dumps(summary),
                mimetype="application/json",
                status_code=200
            )
        else:
            raise Exception(f"OpenSky API returned {response.status_code}")
            
    except Exception as e:
        logging.error(f"Error fetching from OpenSky: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )