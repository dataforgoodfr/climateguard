# Safeguards Infrastructure
This folder contains the framework and infrastructure definition for the safeguards projects. 

## The Stack
In order to deploy the infrastructure we use `brew` and `npm`to install dependencies (`npm` not used now but will be used to pull secrets automatically from Vaultwarden - Install it now !). These need to be installed before running setup.

We deploy using `OpenTofu` which is managed using `tofuenv`. In order to run setup and to run tofu command we use `make` as a wrapper, hence it has to be installed before the . This allows as to source the correct environment variables and to select the correct target directory. Install `make` as well before working with this repository.

## Project Structure
The bulk of the code is defined in the `modules/` directory. here we define the OpenTofu modules that create our infrastructure.  These are separated into `common/` and `country/`. `common/` contains all the infrastructure that is shared for an environment (so for example all the infra shared for the `climate` topic in the `dev/` environment.) and `country` contains the blocks that are specific to a country deployment.

The other folders are `live` folder and a `bin` folder. The `bin` folder contains the setup scripts (and other utility scripts in the future). The `live` folder contains the living infrastructure of the project. It is separated in topics or subjects (only `climate` present at the moment but more are possible). 

Each topic folder has a folder dedicated to the deployment of environments, where each environment is setup as a scaleway project with the name pattern `${subject}-safeguards-${environement}` in a dedicated `.tf` file.

The other folders take on the names of the environments, for example `dev/` contains the living infrastructure for the development environment. The infrastructure is separated in the `folder`, that instantiates the infrastructure from the `common` module, and folders bearing the name of countries, that instantiate different occurrences of the `country` module.

```bash
├── .env.dist
├── .opentofu-version
├── Makefile
├── Readme.md
├── bin
│   ├── setup.sh
│   └── tofu_wrapper.sh
├── live
│   └── climate
│       ├── dev
│       │   ├── common
│       │   │   ├── .env.secrets.dist
│       │   │   ├── backend.tf
│       │   │   ├── main.tf
│       │   │   ├── terraform.tfvars
│       │   │   └── variables.tf
│       │   ├── france
│       │   │   ├── .env.secrets.dist
│       │   │   ├── backend.tf
│       │   │   ├── main.tf
│       │   │   ├── terraform.tfvars
│       │   │   └── variables.tf
│       │   ├── country 2
│       │   │   ├ ...
│       │   │   ├ ...
│       │   .   .
│       │   .   .
│       │   └── country N
│       │       ├ ...
│       │       ├ ...
│       │  
│       └── environments
│           ├── dev.tf
│           ├── main.tf
│           └── prod.tf
└── modules
    ├── common
    │   ├── main.tf
    │   ├── secrets.tf
    │   └── variables.tf
    └── country
        ├── container.tf
        ├── database.tf
        ├── job.tf
        ├── main.tf
        ├── s3.tf
        ├── secrets.tf
        └── variables.tf
```

## Installing dependencies
* `brew` can be installed at [brew.sh](https://brew.sh).
* `brew install make`
* `brew install node`

## Setting up your workspace
In order to setup your workspace you need a Scaleway account, and to be added as a member to the Quotaclimat organisation. From there you can setup you credentials. You will need an API key pair for your account, with the default project selected as the preferred project. The scaleway access key and scaleway secret key need to be added into the `.env` file (which you can create basing yourself on the `.env.dist` file), as `SCW_ACCESS_KEY` and `SCW_SECRET_KEY`. 

Once the `.env` file has been setup, run 
```bash
make setup
```
to install tofu and all dependencies.

In order to deploy our infrastructure, we target individual directories that import the modules into different environments. These environments need different API keys in order to deploy all resources, so create an `.env.secrets` file, based on the `.env.secrets.dist` file provided.
The API keys and rest of the variables in `.env.secret` will be recovered from the [DataForGood Vaultwarden Password Manager](https://vaultwarden.services.dataforgood.fr/). The secrets are available in the collection `Quotaclimat - Désinformation/Terraform/`, and every secret file has the same name as the path in the repository under `infrastructure`: so, for example, the secrets for `live/climate/dev/france/.env.secrets` are in an item named  `live/climate/dev/france` in Vaultwarden and the secrets for `live/climate/dev/common/.env.secrets` are named `live/climate/dev/common` in Vaultwarden. Copy them and paste them in your corresponding local `.env(.secrets)` file.

*TODO: RECOVER THESE DYNAMICALLY FROM VAULTWARDEN*

In order to deploy or modify the infrastructure in an environment, at the very least you need to populate the base `.env` file, the `.env.secrets` file in the `common` folder and the `.env.secrets` file in the country specific folder. So for example you need:
* `.env``
* `live/climate/dev/common/.env.secrets`
* `live/climate/dev/france/.env.secrets`


## Running commands
All commands are ran via `make`. 

In order to run a command, you need to specify the subject (`climate` by default), the environment (`dev` by default) and the country (`common` by default), as the first arguments of the make command. Finally you must run the tofu command separated by a `-`instead of whitespace.
So you will first have to initialise the target folder you are working with using the `make subject=climate env=dev country=france tofu-init` command, and then in a similar manner run `tofu-plan`, `tofu-apply` and `tofu-destroy`. For example to deploy and then destroy the project for the climate topic, on the dev environment for france you would run:
```bash
make subject=climate env=dev country=france tofu-init
make subject=climate env=dev country=france tofu-plan
make subject=climate env=dev country=france tofu-apply
make subject=climate env=dev country=france tofu-destroy
```

In order to deploy changes to the environments (create/destroy a new environment), use the `environments` flag:
```bash
make environments=1 tofu-init
make environments=1 tofu-plan
# etc ..
```

## A note on passwords
When deploying a country for the first time you might need to create passwords for the different users and set tokens. You may use the command 
```bash
openssl rand -base64 16
```
to generate a secure password. (32 chars preferred for the access token.)