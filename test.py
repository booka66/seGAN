import time
from rich.progress import Progress


def test_loader():
    total = 20

    with Progress() as progress:
        task = progress.add_task("[green]Loading...", total=total)

        for i in range(total):
            time.sleep(0.1)
            progress.update(task, advance=1)


if __name__ == "__main__":
    test_loader()
