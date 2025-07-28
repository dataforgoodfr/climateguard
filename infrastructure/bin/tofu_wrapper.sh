#! /bin/bash

tofu_cmd() {
    if [ "$3" == "common" ]; then
        source .env && source live/$1/$2/$3/.env.secrets && tofu -chdir=live/$1/$2/$3 $4
    else
        source .env && source live/$1/$2/$3/.env.secrets && source live/$1/$2/common/.env.secrets && tofu -chdir=live/$1/$2/$3 $4
    fi      
}