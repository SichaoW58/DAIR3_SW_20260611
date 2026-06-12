# -------------------------------------------------------
# DAIR3: Launch the DAIR3 project environment
# Usage: DAIR3
#
# This file assumes that: 
# 1. Your project file is located at $HOME/DAIR3-Workshop
# 		For example: /home/username/DAIR3-Workshop
# 		Edit line 16 to change this
# 2. You ahve a python virtual environment called DAIR3
# 
# On Linux, edit .bashrc and copy this entire file at the end
#	sudo nano ~/.bashrc	
# In nano, after pasting, press CTR + X to exit (select Y on save)
# For mac computers, use .zshrc instead of .bashrc
# -------------------------------------------------------
ALICE() {
    ActivateEnv "DAIR3" "$HOME/DAIR3-Workshop"
}
