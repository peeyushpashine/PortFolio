**Pre-requisites** 

* Ubuntu 18.04+ Linux System 
  * If you do not have a standalone Linux system than it can be installed using your preferred virtualization software [VMWare](https://my.vmware.com/en/web/vmware/info/slug/desktop_end_user_computing/vmware_workstation_pro/15_0) or [Virtual Box](https://www.virtualbox.org/)

  * Install Ubuntu 18 on your virtual machine by
    * downloading Ubuntu image [here](https://www.ubuntu.com/download/desktop)
    * An example of installing Ubuntu using VMWare [here](https://websiteforstudents.com/how-to-install-ubuntu-16-04-17-10-18-04-on-vmware-workstation-guest-machines/)

  * ZMOD application can be installed on your local system (read the fist bullet point for system pre-requisite). In order to install ZMOD on local system, we need docker and docker-compose to be installed. Once you have required system ready, install docker by following instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and docker compose by following instructions [here](https://docs.docker.com/compose/install/). Note than docker-compose should be installed for linux version.

  * Download the docker images tar file to some location on your linux machine and type the below commands:
    
    * docker load --i zmod_nginx.tar
    * docker load --i zmod_zmk.tar
    * docker load --i zmod_zmm.tar


    *Note* :- Try prefix sudo with docker commands if you do not see any results or see an error.
  
  * Check if the docker images are loaded by typing the command:
    * docker image ls 

    [Sample Docker images snapshot from console](https://labcase.softwareag.com/projects/zementis-modeler/repository/revisions/master/entry/Capture.PNG)

  * Once the images are loaded. Start ZMOD application by running the below command. Please make sure that docker-compose.yml and ZMOD.sh lies in the same directory.
    * bash ZMOD.sh

  * Type localhost on your browser(preferably incognito mode/private browsing) you should see the login page. Authenticate using your email/github and you should see **ZMOD** application in action now.


