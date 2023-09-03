
def get_user_input(options):
    user_input = ''
    input_message = "📋 Pick an option:\n"

    for index, item in enumerate(options):
        file = item.split('/')[-1]
        input_message += f'{index+1}) {file}\n'
    input_message += 'Your choice: '
    while user_input not in map(str, range(1, len(options) + 1)):
        user_input = input(input_message)
        
    selected_file = options[int(user_input) - 1]
    print('✅ Your choice is: ' + selected_file.split('/')[-1])
    return selected_file