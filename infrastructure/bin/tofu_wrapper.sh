#! /bin/bash

tofu_cmd() {
    
    if [ "$1" == 1 ]; then
        # if the environments flag is raised
        source .env && tofu -chdir=live/$2/environments $5
    elif [ "$4" == "common" ]; then
        # If working with the common infra, only the common secrets  are retrieved
        source .env && source live/$2/$3/$4/.env.secrets && tofu -chdir=live/$2/$3/$4 $5
    else
        # If working with the country infra, both the common secrets and the country secrets are retrieved
        source .env && source live/$2/$3/$4/.env.secrets && source live/$2/$3/common/.env.secrets && tofu -chdir=live/$2/$3/$4 $5
    fi

}