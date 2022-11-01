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
        "TFY_API_KEY": '<your-api-key>',
        # replace with your run fqn
        "MLF_RUN_FQN": 'truefoundry/akash/movie-clustering-nov-12/cf-model'
    },
)

# replace with your workspace fqn
deployment = service.deploy(workspace_fqn='tfy-cluster-euwe1:akash-test')