"""
How to use:
1) install win32com.client  `pip install pywin32`
2) run the code (outlook must also be open)
3) select our mit account by changing the `account_index` to match order in list
    * order in list changes depending on what is currently open in outlook
4) run the code
5) select the folder with NMR emails by changing `folder_index`
6) enter a `search_key` for your NMR files. Text that is only in the subject line of only NMR emails
7) enter a `save_location`
8) run the code

Microsoft Reference for COM objects:
https://learn.microsoft.com/en-us/dotnet/api/microsoft.office.interop.outlook.namespace?view=outlook-pia

https://stackoverflow.com/questions/22813814/clearly-documented-reading-of-emails-functionality-with-python-win32com-outlook
"""

import os
import shutil
import zipfile
import pathlib

import win32com.client  # `pip install pywin32`

# parameters
account_index = 3
folder_index = 5
search_key = "DW2-14"  # key word in subject line of ONLY NMR files
save_location = r"C:\Users\nicep\Desktop\DW2_14"
remove_extra_folder_layers = True


def main():
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

    # get accounts
    folders = outlook.Folders
    print("email accounts")
    for i, folder in enumerate(folders):
        print(i, folder.Name)

    account = folders[account_index]  # << change index

    # get folder
    print("\nemail account folders")
    for i, folder in enumerate(account.Folders):
        print(i, folder.Name)
    nmr_data_folder = account.Folders[folder_index]

    # get email
    emails = [email for email in nmr_data_folder.Items if search_key in email.Subject]

    # get attachments
    attachments = [email.Attachments[0] for email in emails]  # only takes first attachment in email

    # save attachments
    path = pathlib.Path(save_location)

    if not os.path.exists(path):
        os.makedirs(path)

    print("\nsaving:", len(attachments), "attachments")
    for attachment in attachments:
        # save zip folder
        file_location = path / attachment.FileName
        if file_location.suffix != ".zip":
            continue
        attachment.SaveAsFile(file_location)

        # unzip folder
        with zipfile.ZipFile(file_location, 'r') as zip_ref:
            new_path = path / attachment.FileName.strip(".zip")
            zip_ref.extractall(new_path)

        # remove one layer
        if remove_extra_folder_layers:
            sub_folder = find_folder(new_path)
            sub_sub_folder = find_folder(sub_folder)
            shutil.move(sub_sub_folder, sub_sub_folder.parent.parent.parent)
            shutil.rmtree(new_path, ignore_errors=False)

        # delete unzip folder
        os.remove(file_location)


def find_folder(root_folder: str | pathlib.Path, target_folder_name: str = None) -> pathlib.Path:
    if not os.path.isdir(root_folder):
        raise ValueError(f"{root_folder} is not a valid directory.")

    contents = os.listdir(root_folder)

    for entry in contents:
        entry_path = os.path.join(root_folder, entry)

        if os.path.isdir(entry_path):
            if target_folder_name is None:
                return pathlib.Path(entry_path)
            if entry == target_folder_name:
                return pathlib.Path(entry_path)

    if target_folder_name is None:
        raise ValueError(f"No folders found in {root_folder}.")
    raise ValueError(f"Folder '{target_folder_name}' not found in {root_folder}.")


if __name__ == "__main__":
    main()
    print("done")
