import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_structure(s):
    direction_map = {"n": (0, 1), "s": (0, -1), "e": (1, 0), "w": (-1, 0)}

    commands = s.strip().split("\n")

    positions = {}

    first_obj = True

    for command in commands:
        if command.split(" ")[0] in ["add", "remove"]:
            parts = command.split(" ")

            if len(parts) == 4 and parts[0] == "add":
                from_blob, to_blob, direction = parts[1], parts[2], parts[3][1:-1]

                if first_obj:
                    positions[from_blob] = (0, 0)
                    first_obj = False

                dx, dy = direction_map[direction]
                new_x = positions[from_blob][0] + dx
                new_y = positions[from_blob][1] + dy

                positions[to_blob] = (new_x, new_y)

            elif len(parts) == 2 and parts[0] == "remove":
                blob = parts[1]

                if blob in positions:
                    del positions[blob]
                else:
                    print(f"Invalid command: {command}. Blob does not exist.")
            else:
                print(f"Invalid command: {command}")

    fig, ax = plt.subplots()
    for blob, (x, y) in positions.items():
        ax.add_patch(
            patches.Rectangle(
                (x, y), 1, 1, linewidth=1, edgecolor="blue", facecolor="blue"
            )
        )
        plt.text(x + 0.5, y + 0.5, blob, ha="center", va="center", color="white")

    ax.axis("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.show()
