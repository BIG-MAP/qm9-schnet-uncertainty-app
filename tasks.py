from invoke import task

from settings import DOCKER_TAG, BIOLIB_TOKEN, BIOLIB_USERNAME, BIOLIB_APPNAME


@task
def build(c):
    c.run(f"docker buildx build -t {DOCKER_TAG} . --platform linux/amd64")


@task
def push(c):
    with c.cd('biolib'):
        c.run(
            f"biolib push {BIOLIB_USERNAME}/{BIOLIB_APPNAME}",
            env={"BIOLIB_TOKEN": BIOLIB_TOKEN}
        )
