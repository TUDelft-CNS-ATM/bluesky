# How to install BlueSky in Fedora env

1) Install Anaconda

2) create anaconda repo

    conda create --name bluesky_fedora

3) Init anaconda env

    conda activate bluesky_fedora

4) Install Bluesky dependencies

    cd bluesky/
    pip install -e ".[full]"


# How to Configure BlueSky


1) Configuration file (simu-config.json)

Adjust the simu-config-json with the desired params of the simulation:

- simulation_time_sec
- incremental_time_sec
- scenario_file_name "DEMO/demo-scenario.scn


2) Start the simulation in different terminals

    python BlueSky.py

    python orchestrator.py


3) At the end the Log file will be saved:

The log files are saved in CSV format inside the bluesky/log folder.



# Common errors

1) If you find this message: [BlueSky-Sim] Orchestrator socket disabled: [Errno 98] Address already in use

- check the pid process: sudo lsof -i :12000

- kill the return port: sudo kill <pid>x