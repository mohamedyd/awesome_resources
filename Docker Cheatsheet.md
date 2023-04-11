In this doument, I list the important resources and commands for learning containers.

| | |
|-|-|
|Bash Command|Details|
|```Sudo getent group```|The `getent` command is used to retrieve information from various databases on a system, including the system password database, group database, hostname database, and service name database. `getent group` will output a list of all groups on the system, along with their associated group IDs and member users.|
|```Sudo usermod -aG docker sag```| It adds the user "sag" to the docker group on a Linux system, giving them permission to run Docker commands without using sudo. `usermod`: The usermod command is used to modify user account details on a Linux system. `-aG docker`: The `-a` option specifies that we want to add the user to a group, rather than replacing their current group memberships. The `G` option specifies the name of the group we want to add the user to, which in this case is docker.|


| | |
|-|-|
|Docker Commands|Details|
|```Docker info```|It provides a summary of the Docker daemon's configuration, network settings, and various system-wide settings.|


| | |
|-|-|
|Docker Image Commands|Details|
|`Docker image pull fedora`|Download a Docker image of the Fedora Linux operating system from the official Docker Hub repository.|
|`Docker image ls`|List the downloaded images|
|`Docker image rm emb`|Remove the image whose ID starts with "emb"|
|`Docker image inspect ubuntu`|Retrieve detailed information about the Docker image "Ubuntu". It can be used to inspect a single image or multiple images at once.|
|`Docker image ls -q`|Retrieve the IDs of all available images|
|`Docker image rm $(Docker image ls -q)`|Remove all the available images|


| | |
|-|-|
|Docker Container Commands|Details|
|`Docker container create -it alpine bash`|It creates a new Docker container based on the Alpine Linux image, and launches a Bash shell inside the container. `-it`: Allocates a pseudo-TTY and opens an interactive terminal session to the container, allowing you to type commands and see their output in real-time.|
|`Docker container ls`|List the running containers|
|`Docker container ls -a `|List all available containers (`-a` is equivalent to `--all`)|
|`Docker container run -it python`|It creates a new Docker container based on the official Python image, and launches an interactive terminal session inside the container. If the Python image does not exist on your system, the command also pull it from Docker Hub. In this case, the main command, whose PID=1, is `python3`. If python3 is exited, the entire container will be terminated|
|`Docker container run -it python /bin/bash`|It creates a new Docker container based on the official Python image, and launches an interactive Bash shell inside the container. In this case, the main command, whose PID=1, is `bash`. If bash is exited, the entire container will be terminated. We can work with other applications from bash, for example, `python3`. If python3 is exited, the container will not be terminated since it is not the main command.|
|`Docker container rm e8t`|Remove the container whose ID starts with "e8t"|
|`docker container run -it --name "hadoopc" -h hadoopc 76/hadoop3:latest bash -c "/user/local/booststrap.sh; bash"`|The command runs a container using the image "76/hadoop3:latest" with the name "hadoopc" and hostname "hadoopc". The container is started in interactive mode using the `-it` flag, which attaches the container to your terminal and allows you to interact with it. The command executed inside the container is "/user/local/booststrap.sh; bash", which runs a script called `bootstrap.sh` located in the /user/local directory, and then starts a new bash shell. The purpose of the bootstrap script is to prepare the environment for the application or service that will run inside the container.|
|`Docker container stop`|Stop the container|



# Resources

* [Big Data Courses (Arabic)](https://github.com/ahmedsami76/AraBigData): long videos covering various topics, such as [Docker](https://github.com/ahmedsami76/AraBigData/blob/main/Docker.ipynb), Git, and Hadoop. [[Youtube Channel](https://www.youtube.com/@bigdata4756)]