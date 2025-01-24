from invoke import task

@task
def echo(c, msg: str):
    c.run(f"echo {msg}")