#!/bin/bash
cd /home/$1 &&\
commit_id=ea1445cc7016315d0f5728f8e8b12a45dc0a7286 &&\
curl -sSL "https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable" -o vscode-server-linux-x64.tar.gz &&\
mkdir -p ~/.vscode-server/bin/${commit_id} &&\
tar zxvf vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/${commit_id} --strip 1 &&\
touch ~/.vscode-server/bin/${commit_id}/0 &&\
# install vscode extensions.
/home/$1/.vscode-server/bin/${commit_id}/bin/code-server \
--install-extension ms-python.python \
--install-extension ms-toolsai.jupyter