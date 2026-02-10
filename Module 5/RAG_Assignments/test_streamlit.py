import streamlit as sl

sl.title('First sl app..')
sl.write('Display of paragraph')

user_input = sl.text_input('Name: ')

if user_input:
    sl.write(f'Hello: {user_input}, welcome onboard')