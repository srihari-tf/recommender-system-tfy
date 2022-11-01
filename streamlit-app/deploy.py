from servicefoundry import Build, Service, DockerFileBuild


service = Service(
    name="movie-rec-app",
    image=Build(
        build_spec=DockerFileBuild(
            dockerfile_path='Dockerfile',
            build_context_path='./',
        )
    ),
    ports=[{"port": 8080}],
    env={
        "TFY_API_KEY": 'djE6dHJ1ZWZvdW5kcnk6YWthc2g6YTQwZTIx',
        "MLF_RUN_FQN": 'truefoundry/akash/movie-clustering-nov-12/cf-model'
    },
)
deployment = service.deploy(workspace_fqn='tfy-cluster-euwe1:akash-test')