# Safeguards Infrastructure
This folder contains the framework and definition for the safeguards projects. 

## The Stack
In order to deploy the infrastructure we use `brew` and `npm`to install dependencies (`npm` not used now but will be used to pull secrets automatically from vaultwarden). These need to be installed before running setup.

We deploy using `OpenTofu` which is managed using `tofuenv`. In order to run setup and to run tofu command we use `make` as a wrapper, hence it has to be installed before the . This allows as to source the correct environment variables and to select the correct target directory. Install `make` as well before working with this repository.

## Project Structure
The bulk of the code is defined in the `modules/` directory. here we define the OpenTofu modules that create our infrastructure.  These are separated into `common/` and `country/`. `common/` contains all the infrastructure that is shared for an environment (so for example all the infra shared for the `climate` topic in the `dev/` environment.) and `country` contains the blocks that are specific to a country deployment.

The other folders are `live` folder and a `bin` folder. The `bin` folder contains the setup scripts (and other utility scripts in the future). The `live` folder contains the living infrastructure of the project. It is separated in topics or subjects (only `climate` present at the moment but more are possible). 

Each topic folder has a folder dedicated to the deployment of environments, where each environment is setup as a scaleway project with the name pattern `${subject}-safeguards-${environement}` in a dedicated `.tf` file.

The other folders take on the names of the environments, for example `dev/` contains the living infrastructure for the development environment. The infrastructure is separated in the `folder`, that instantiates the infrastructure from the `common` module, and folders bearing the name of countries, that instantiate different occurrences of the `country` module.

```bash
├── Makefile
├── Readme.md
├── bin
│   └── setup.sh
├── live
│   └── climate
│       ├── dev
│       │   ├── common
│       │   │   ├── backend.tf
│       │   │   ├── common.tf
│       │   │   ├── providers.tf
│       │   │   ├── terraform.tfvars
│       │   │   └── variables.tf
│       │   └── france
│       │       ├── backend.tf
│       │       ├── main.tf
│       │       ├── providers.tf
│       │       ├── terraform.tfvars
│       │       └── variables.tf
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
In order to setup your workspace you need a Scaleway account, and to be added as a member to the Quotaclimat organisation. From there you can setup you credentials. You will need:
* An API key pair for the default accounts.
* An API key pair for each environment you want to work in (dev/prod)

The API key for default will be placed in a .env file next to the make file following the configuration in `.env.dist`. Once `.env` file has been created, run 
```bash
make setup
```
to install tofu and all dependencies.

In order to deploy we target individual directories that import the modules into different environments. These environments need different API keys in order to deploy all resources, so create an `.env.secrets` file, based on the `.env.secrets.dist` file provided, where you add your access and secret key for the environment in question. 

*TODO: CREATE GENERAL CREDENTIALS FOR TERRAFORM FOR EACH ENVIRONMENT, AND PULL THEM FROM VAULTWARDEN.*

**THE ONLY EXCEPTION WILL BE THE `environments/` folder here you can use the `.env.secrets.dist` file as is (you just have to create a copy named `.env.secrets`)**

The rest of the variables in `.env.secret` will be recovered from the [DataForGood Vaultwarden Password Manager](https://vaultwarden.services.dataforgood.fr/). The secrets are available in the collection `Quotaclimat - Désinformation/Terraform/`, and every secret file has the same name as the path in the repository under `infrastructure`: so, for example, the secrets for `live/climate/dev/france/.env.secrets` are in an item named  `live/climate/dev/france` in Vaultwarden. 

*TODO: RECOVER THESE DYNAMICALLY FROM VAULTWARDEN*

## Running commands
All commands are run via `make`. So you will first have to initialise the target folder you are working with using the `make tofu_dir=path/to/dir tofu-init` command, and then in a similar manner run `tofu-plan`, `tofu-apply` and `tofu-destroy`. For example to deploy and then destroy the project for the climate topic, on the dev environment for france you would run:
```bash
make tofu_dir=live/climate/dev/france tofu-init
make tofu_dir=live/climate/dev/france tofu-plan
make tofu_dir=live/climate/dev/france tofu-apply
make tofu_dir=live/climate/dev/france tofu-destroy
```
