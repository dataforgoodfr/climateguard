init:
	source $(tofu_dir)/.env && tofu -chdir=$(tofu_dir) init

plan:
	source $(tofu_dir)/.env && tofu -chdir=$(tofu_dir) plan

apply:
	source $(tofu_dir)/.env && tofu -chdir=$(tofu_dir) apply

fmt:
	source $(tofu_dir)/.env && tofu -chdir=$(tofu_dir) fmt

get-state:
	source $(tofu_dir)/.env && tofu -chdir=$(tofu_dir) state pull

state:
	source $(tofu_dir)/.env && tofu -chdir=$(tofu_dir) state list

destroy:
	source $(tofu_dir)/.env && tofu -chdir=$(tofu_dir) destroy