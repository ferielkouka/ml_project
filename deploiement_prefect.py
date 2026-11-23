from prefect import serve
from prefect.client.schemas.schedules import CronSchedule
from pipeline_prefect import flow_all, flow_train, flow_evaluate, flow_code

TOUS_LES_JOURS_2H = [CronSchedule(cron="0 2 * * *", timezone="Africa/Tunis")]

if __name__ == "__main__":
    serve(
        flow_all.to_deployment(name="ml-pipeline-all", schedules=TOUS_LES_JOURS_2H, tags=["mlops", "full-pipeline"]),
        flow_train.to_deployment(name="ml-pipeline-train", schedules=TOUS_LES_JOURS_2H, tags=["mlops", "train"]),
        flow_evaluate.to_deployment(name="ml-pipeline-evaluate", tags=["mlops", "evaluate"]),
        flow_code.to_deployment(name="ml-pipeline-code", tags=["mlops", "code"]),
    )
