# Define tofu_dir if not passed in externally
tofu_dir ?= .

# Using a recipe macro to improve DRY
define TOFU_CMD
	@bash -c 'source $(tofu_dir)/.env && tofu -chdir=$(tofu_dir) $(1)'
endef

.PHONY: setup tofu-init tofu-plan tofu-apply tofu-fmt tofu-state-pull tofu-state-list tofu-destroy

setup:
	@bash ./bin/setup.sh

tofu-init:
	$(call TOFU_CMD,init)

tofu-plan:
	$(call TOFU_CMD,plan)

tofu-apply:
	$(call TOFU_CMD,apply)

tofu-fmt:
	$(call TOFU_CMD,fmt)

tofu-state-pull:
	$(call TOFU_CMD,"state pull")

tofu-state-list:
	$(call TOFU_CMD,"state list")

tofu-destroy:
	$(call TOFU_CMD,destroy)
