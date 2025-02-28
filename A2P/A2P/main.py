from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import urllib.parse
import os


from sqlalchemy import create_engine
from contextlib import asynccontextmanager
from sqlalchemy.orm import sessionmaker
from fastapi import Depends

import datetime
import logging
import time
import psutil
import json
import torch
from collections import deque

from custom_logger.logger_manager import configuelogging

logger = configuelogging()

with open(os.path.join(os.path.dirname(__file__), 'config', 'db_config.json'), 'r') as file:
    db_config = json.load(file)


with open(os.path.join(os.path.dirname(__file__), 'config', 'app_config.json'), 'r') as file:
    app_config = json.load(file)

try:

    host = db_config.get("host")
    port = db_config.get("port")
    user = db_config.get("user")
    password = db_config.get("password")
    database = db_config.get("database")

    # host = "192.168.0.127"
    # port = 3306
    # user = "root"
    # password = "Omobio@123"
    # database = "firewall"
    encoded_password = urllib.parse.quote_plus(password)

    db_uri = f"mysql+pymysql://{user}:{encoded_password}@{host}:{port}/{database}"
    print(db_uri)
    db_uri_async = f"mysql+aiomysql://{user}:{encoded_password}@{host}:{port}/{database}"
    engine = create_engine(db_uri)

    # Create the SQLAlchemy engines
    engine = create_engine(db_uri)
    async_engine = create_async_engine(db_uri_async)

    # AsyncSessionLocal = sessionmaker(
    #     async_engine,
    #     expire_on_commit=False,
    #     class_=AsyncSession
    # )

except Exception as e:

    logger.exception("An error occurred: %s", e)


lock = asyncio.Lock()
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    global model, db_embeddings, block_arr, operator_order_arr, operator_df, operator_detect_dict, operator_auto_train_dict
    global manual_emb_df, auto_emb_df, emb_df, operator_df
    global auto_emb_df_update_cutoff, in_memory_df_refresh_interval, auto_emb_db_sync_interval
    auto_emb_df_update_cutoff = app_config.get("auto_emb_df_update_cutoff")
    in_memory_df_refresh_interval = app_config.get(
        "in_memory_df_refresh_interval")
    auto_emb_db_sync_interval = app_config.get("auto_emb_db_sync_interval")

    # model = SentenceTransformer(
    #     'sentence-transformers/all-MiniLM-L6-v2', device='cpu:0')

    standard_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    standard_model.to('cpu')  # Ensure model is on CPU for quantization
    model = torch.quantization.quantize_dynamic(
        standard_model, {torch.nn.Linear}, dtype=torch.qint8
    )


    try:
        initialize_startup_variables()
        scheduler.add_job(refresh_variable_on_in_memory_db_update, 'interval',
                          seconds=in_memory_df_refresh_interval)
        scheduler.add_job(sync_auto_emb_df_with_db, 'interval',
                          seconds=auto_emb_db_sync_interval)
        scheduler.start()

        logger.info("Application initialized successfully")

        yield  # This is the point separating startup from shutdown logic
    finally:
        # Begin shutdown and resource cleanup

        scheduler.shutdown()

        del model
        logger.info("Application shutdown successfully")


app = FastAPI(lifespan=app_lifespan)

# APi Endpoints

# # Global variable to store last 5 total_time values
# last_5_total_times = deque(maxlen=5)

# @app.post("/compare")
# async def compare_content(request: Request, background_tasks: BackgroundTasks):
#     global last_5_total_times
#     start_time = time.time()
    
#     try:
#         # Parse JSON data from the request
#         json_data = await request.json()
#         json_data_time = time.time()

#         msg_content = json_data['Content']
#         operator_name = int(json_data["OperatorId"])

#         start_emb_time = time.time()
#         given_embedding = model.encode(msg_content, convert_to_numpy=True)
#         end_emb_time = time.time()

#         block_label, block_index = content_processing(
#             msg_content, given_embedding, operator_name, db_embeddings,
#             block_arr, operator_detect_dict, operator_auto_train_dict, 
#             operator_order_arr, background_tasks
#         )

#         end_time = time.time()
#         total_time = (end_time - start_time) * 1000  # Convert to milliseconds

#         # Store total_time in deque (maintains last 5 values)
#         last_5_total_times.append(total_time)

#         # Compute average of last 5 messages
#         avg_total_time = sum(last_5_total_times) / len(last_5_total_times)

#         log_message = (
#             f"JSON Read Time: {-(start_time-json_data_time)*1000:.2f} ms, "
#             f"Embedding Time: {-(start_emb_time-end_emb_time)*1000:.2f} ms, "
#             f"Numpy Operation Time: {-(end_emb_time-end_time)*1000:.2f} ms, "
#             f"Computation Time: {-(start_emb_time-end_time)*1000:.2f} ms, "
#             f"Compare Total Time: {total_time:.2f} ms, "
#             f"Avg Total Time (Last 5): {avg_total_time:.2f} ms"
#         )

#         logger.info(log_message)

#         if block_label == 'Pass':
#             logger.info(f'Passed the content - {msg_content} - Operator {operator_name}')
#             return '0'
#         else:
#             logger.info(f'Blocked the content - {msg_content} - Operator {operator_name}')
#             return '1'

#     except Exception as e:
#         logger.error(f"Error: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

@app.post("/compare")
async def compare_content(request: Request, background_tasks: BackgroundTasks):
    start_time = time.time()
    try:
        # Parse JSON data from the request
        json_data = await request.json()
        json_data_time = time.time()

        msg_content = json_data['Content']
        operator_name = int(json_data["OperatorId"])

        start_emb_time = time.time()

        given_embedding = model.encode(msg_content, convert_to_numpy=True)
        end_emb_time = time.time()

        block_label, block_index = content_processing(msg_content, given_embedding, operator_name, db_embeddings,
                                                      block_arr, operator_detect_dict, operator_auto_train_dict, operator_order_arr, background_tasks)

        end_time = time.time()
        json_read_time = -(start_time-json_data_time)*1000
        embedding_time = -(start_emb_time-end_emb_time)*1000
        numpy_operation_time = -(end_emb_time-end_time)*1000
        computation_time = -(start_emb_time-end_time)*1000
        total_time = -(start_time-end_time)*1000

        log_message = (
            f"JSON Read Time: {json_read_time:.2f} ms, "
            f"Embedding Time: {embedding_time:.2f} ms, "
            f"Numpy Operation Time: {numpy_operation_time:.2f} ms, "
            f"Computation Time: {computation_time:.2f} ms, "
            f"Compare Total Time: {total_time:.2f} ms"
        )

        logger.info(log_message)

        if block_label == 'Pass':
            logger.info(
                f'Passed the content - {msg_content} - Operator {operator_name}')
            return '0'
        else:
            logger.info(
                f'Blocked the content - {msg_content} - Operator {operator_name}')
            return '1'

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/operator_update")
async def operator_update(request: Request, background_tasks: BackgroundTasks):
    global auto_emb_df, operator_df, manual_emb_df, db_embeddings
    try:

        json_data = await request.json()

        background_tasks.add_task(
            refresh_variable_on_db_update)
        logger.info(f"DB update request recived")

        return {"Status": "success"}

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Compare Endpoints Calculation Fucntions


def content_processing(msg_content, given_embedding, operator_name, db_embeddings, block_arr, operator_detect_dict, operator_auto_train_dict, operator_order_arr, background_tasks):

    is_a_operator, detect_value, auto_train_value = check_operator_in_operator_df(
        operator_name, operator_detect_dict, operator_auto_train_dict)

    if not is_a_operator:

        block_label, block_index = find_similarity_non_db_operator(
            given_embedding, db_embeddings, block_arr)

    else:
        if detect_value == 0:
            block_label, block_index = "Pass", -1
            if auto_train_value == 1:
                background_tasks.add_task(update_in_memory_auto_emb_df, msg_content, db_embeddings,
                                          given_embedding, operator_order_arr, auto_emb_df_update_cutoff, operator_name)

        else:
            block_label, block_index = find_similarity_in_db_operator(
                given_embedding, db_embeddings, block_arr, operator_order_arr, operator_name)

            if block_label == "Pass" and auto_train_value == 1:
                background_tasks.add_task(update_in_memory_auto_emb_df, msg_content, db_embeddings,
                                          given_embedding, operator_order_arr, auto_emb_df_update_cutoff, operator_name)

    return block_label, block_index


def check_operator_in_operator_df(operator_name, operator_detect_dict, operator_auto_train_dict):
    is_found = operator_name in operator_detect_dict
    detect_value = operator_detect_dict.get(operator_name, None)
    auto_train_value = operator_auto_train_dict.get(operator_name, None)

    return is_found, detect_value, auto_train_value


def find_similarity_in_db_operator(given_embedding, db_embeddings, block_arr, operator_order_arr, operator_name):
    dot_products = np.dot(db_embeddings, given_embedding)
    if np.all(operator_order_arr[np.where(dot_products > block_arr)[0]] == operator_name):
        return "Pass", -1
    else:
        block_indices = np.where(dot_products > block_arr)[0]
        filtered_operators = operator_order_arr[block_indices]
        filtered_indices = block_indices[filtered_operators != operator_name]
        block_index = filtered_indices[np.argmax(
            dot_products[filtered_indices])]
        return "Block", block_index


def find_similarity_non_db_operator(given_embedding, db_embeddings, block_arr):
    # Compute dot products
    dot_products = np.dot(db_embeddings, given_embedding)

    # Check if any value meets the block threshold
    is_block = np.any(dot_products > block_arr)
    if is_block:
        block_index = np.argmax(dot_products > block_arr)
        return "Block", block_index

    return "Pass", -1


# Update In Memory auto_emb_db Functions


async def update_in_memory_auto_emb_df(msg_content, db_embeddings, given_embedding, operator_order_arr, auto_emb_df_update_cutoff, operator_name):
    global auto_emb_df

    if check_auto_emb_df_update_requirement(db_embeddings, given_embedding, operator_order_arr, auto_emb_df_update_cutoff, operator_name):
        new_auto_emb_df = append_to_auto_emb_df(
            operator_name, msg_content, given_embedding, auto_emb_df, auto_emb_df_update_cutoff)
        async with lock:
            auto_emb_df = new_auto_emb_df
        logger.info("In memory auto_emb_df updated")


def check_auto_emb_df_update_requirement(db_embeddings, given_embedding, operator_order_arr, auto_emb_df_update_cutoff, operator_name):
    # Calculate dot products
    dot_products = np.dot(db_embeddings, given_embedding)

    # Find indices where the operator name matches
    eligible_indices = np.where(operator_order_arr == operator_name)[0]

    # Select dot products based on eligible indices
    selected_dot_products = dot_products[eligible_indices]

    # Check if all selected dot products are below the cutoff
    is_auto_emb_df_update_required = np.all(
        selected_dot_products < auto_emb_df_update_cutoff)

    return is_auto_emb_df_update_required


def append_to_auto_emb_df(operator_name, msg_content, given_embedding, auto_emb_df, auto_emb_df_update_cutoff):

    filtered_auto_emb_df = auto_emb_df[(
        auto_emb_df['operator_name'] == operator_name) & (auto_emb_df['is_in_db'] == 0)]
    if not filtered_auto_emb_df.empty:
        temp_db_embeddings = np.vstack(
            filtered_auto_emb_df['content_emb'].values)

        dot_products = np.dot(temp_db_embeddings, given_embedding)

        is_auto_emb_df_update_required = np.all(
            dot_products < auto_emb_df_update_cutoff)
        if is_auto_emb_df_update_required:

            new_row = {
                'record_id': -1,
                'operator_name': operator_name,
                'content': msg_content,
                'content_emb': given_embedding,
                'is_in_db': 0
            }
            new_row_df = pd.DataFrame([new_row])
            auto_emb_df = pd.concat(
                [auto_emb_df, new_row_df], ignore_index=True)

    else:

        new_row = {
            'record_id': -1,
            'operator_name': operator_name,
            'content': msg_content,
            'content_emb': given_embedding,
            'is_in_db': 0
        }
        new_row_df = pd.DataFrame([new_row])
        auto_emb_df = pd.concat([auto_emb_df, new_row_df], ignore_index=True)

    return auto_emb_df


# Refresh In Memory Variable Fuctions


async def refresh_variable_on_in_memory_db_update():
    global db_embeddings, block_arr, operator_order_arr, auto_emb_df, operator_df, manual_emb_df

    # Create the embedding dataframe
    emb_df = emb_df_from_auto_manual_emb_dfs(auto_emb_df, manual_emb_df)

    # Create db embeddings
    new_db_embeddings, new_block_arr, new_operator_order_arr = create_db_embeddings(
        emb_df, operator_df)

    # Acquire lock before updating global variables
    async with lock:
        db_embeddings = new_db_embeddings
        block_arr = new_block_arr
        operator_order_arr = new_operator_order_arr

    logger.info("Refresh variable on in memory update")


async def refresh_variable_on_db_update():

    global auto_emb_df, db_embeddings, block_arr, operator_order_arr, operator_df, operator_detect_dict, operator_auto_train_dict, manual_emb_df

    new_operator_df = make_operator_df()

    new_operator_detect_dict, new_operator_auto_train_dict, new_operator_manual_train_dict = make_dicts_from_operator_df(
        new_operator_df)

    new_manual_emb_df = get_manual_emb_df_from_db(new_operator_df)

    emb_df = emb_df_from_auto_manual_emb_dfs(auto_emb_df, new_manual_emb_df)

    new_db_embeddings, new_block_arr, new_operator_order_arr = create_db_embeddings(
        emb_df, operator_df)

    async with lock:
        db_embeddings = new_db_embeddings
        block_arr = new_block_arr
        operator_order_arr = new_operator_order_arr
        operator_df = new_operator_df
        operator_detect_dict = new_operator_detect_dict
        operator_auto_train_dict = new_operator_auto_train_dict
        manual_emb_df = new_manual_emb_df

    logger.info("Refresh variable on DB update")


# Sync auto_emb_df with the DB


async def sync_auto_emb_df_with_db():
    global auto_emb_df
    filtered_df = auto_emb_df[auto_emb_df['is_in_db'] == 0]
    
    async_engine = create_async_engine(db_uri_async)
    AsyncSessionLocal = sessionmaker(
        async_engine,
        expire_on_commit=False,
        class_=AsyncSession
)


    async with AsyncSessionLocal() as session:
        for index, row in filtered_df.iterrows():
            try:
                # Prepare the INSERT statement using text()
                insert_query = text("""
                INSERT INTO operator_embeddings (operator_name, content, content_emb) 
                VALUES (:operator_name, :content, :content_emb)
                """)
                # Data to be inserted
                data = {"operator_name": row['operator_name'],
                        "content": row['content'],
                        "content_emb": row['content_emb'].tobytes()}

                # Execute the query asynchronously
                await session.execute(insert_query, data)
                await session.commit()
                logger.info(
                    f"Auto train DB record inserted for operator: {row['operator_name']}")

                # Update the is_in_db column in the global DataFrame
                async with lock:
                    auto_emb_df.at[index, 'is_in_db'] = 1

            except Exception as e:
                logger.error(f"Error: {e}")


# Fuctions for get data from DB and create in memory variables from them

def make_operator_df():
    print('make_operator_df function called')

    try:

        with engine.connect() as conn:

            tbl_operator_rule_query = """

                SELECT *
                FROM tbl_operator_rule
                WHERE rule IN (SELECT id FROM tbl_rules_config WHERE rule LIKE 'ai') 
                


            """

            tbl_operator_rule_df = pd.read_sql_query(tbl_operator_rule_query, conn)
            tbl_global_rule_query = """

            SELECT * FROM tbl_rule WHERE rule IN (SELECT id FROM tbl_rules_config WHERE rule LIKE 'ai');
                
            """

            tbl_global_rule_df = pd.read_sql_query(tbl_global_rule_query, conn)
            operator_rule_settings_query = """
            SELECT * 
            FROM `tbl_rule_setting` 
            WHERE `rule` IN (
                SELECT id
                FROM tbl_operator_rule
                WHERE rule IN (SELECT id FROM tbl_rules_config WHERE rule LIKE 'ai') 
                
                
            )
            AND `tbl` = 'tbl_operator_rule';
            """

            # Execute the query and load into DataFrame

            operator_rule_settings_df = pd.read_sql_query(
                operator_rule_settings_query, conn)

            global_rule_settings_query = """
            SELECT *  FROM `tbl_rule_setting` WHERE `rule` in (SELECT id FROM tbl_rule WHERE rule IN (SELECT id FROM tbl_rules_config WHERE rule LIKE 'ai'))AND tbl = 'tbl_rule'
            """

            # Execute the query and load into DataFrame

            global_rule_settings_df = pd.read_sql_query(
                global_rule_settings_query, conn)

            tbl_operator_rule_df['global_bypass'] = tbl_global_rule_df['bypass'].iloc[0]
            tbl_operator_rule_df.rename(
                columns={'bypass': 'operator_bypass'}, inplace=True)

            def determine_rule_owner(row):
                if row['operator_bypass'] == 0:
                    return 'operator'
                elif row['operator_bypass'] == 1 and row['global_bypass'] == 0:
                    return 'global'
                else:
                    return 'none'
            tbl_operator_rule_df['rule_owner'] = tbl_operator_rule_df.apply(
                determine_rule_owner, axis=1)

            filtered_operator_df = tbl_operator_rule_df[tbl_operator_rule_df['rule_owner'] != 'none']

            # Initialize a list to store the new rows
            new_rows = []

            try:
                # Iterate over the filtered rows
                for idx, row in filtered_operator_df.iterrows():
                    operator = row['operator']
                    rule_id = row['id']
                    rule_owner = row['rule_owner']

                    if rule_owner == 'operator':
                        # Get settings from operator_rule_settings_df
                        settings_df = operator_rule_settings_df[operator_rule_settings_df['rule'] == rule_id]
                    elif rule_owner == 'global':
                        # Use global_rule_settings_df
                        settings_df = global_rule_settings_df

                    # Extract required values
                    auto_train = settings_df[settings_df['name']
                                            == 'auto_train']['value'].values[0]
                    detect = settings_df[settings_df['name']
                                        == 'detect']['value'].values[0]
                    detect_score = settings_df[settings_df['name']
                                            == 'detect_score']['value'].values[0]
                    manual_train = settings_df[settings_df['name']
                                            == 'manual_train']['value'].values[0]

                    # Append to the new_rows list
                    new_rows.append({'operator_name': operator, 'auto_train': auto_train,
                                    'manual_train': manual_train, 'detect': detect, 'block_score': detect_score, })
                    
            except:
                pass

            # Create new DataFrame

            default_row = {
                'operator_name': [10000],
                'auto_train': [0],
                'manual_train': [1],
                'detect': [0],
                'block_score': [99]
            }

            operator_df = pd.DataFrame(default_row)
            new_rows_df = pd.DataFrame(new_rows)
            operator_df = pd.concat(
                [operator_df, new_rows_df], ignore_index=True)
            operator_df['operator_name'] = operator_df['operator_name'].astype(
                int)
            operator_df['auto_train'] = operator_df['auto_train'].astype(int)
            operator_df['manual_train'] = operator_df['manual_train'].astype(
                int)
            operator_df['detect'] = operator_df['detect'].astype(int)

            # Convert the last column to float
            operator_df['block_score'] = operator_df['block_score'].astype(
                float)
            operator_df['block_score'] = operator_df['block_score']/100

        return operator_df
    except Exception as e:
        print('connection failed')
        logger.error(f"Error: {e}")


def make_dicts_from_operator_df(operator_df):
    # print('make_dicts_from_operator_df function called')

    operator_detect_dict = dict(
        zip(operator_df['operator_name'], operator_df['detect']))
    operator_auto_train_dict = dict(
        zip(operator_df['operator_name'], operator_df['auto_train']))
    operator_manual_train_dict = dict(
        zip(operator_df['operator_name'], operator_df['manual_train']))

    return operator_detect_dict, operator_auto_train_dict, operator_manual_train_dict


def get_manual_emb_df_from_db(operator_df):
    # Read table from MySQL using the engine
    manual_emb_query = "SELECT * FROM tbl_ai_a2p_content;"
    manual_emb_df = pd.read_sql_query(manual_emb_query, engine)

    # Drop unnecessary columns
    manual_emb_df.drop(columns=['created_ts', 'updated_ts'], inplace=True)

    # Processing each row
    for index, row in manual_emb_df.iterrows():
        content_emb = row['content_emb']

        if content_emb is None:
            new_content_emb = model.encode(
                row['content'], convert_to_numpy=True)
            manual_emb_df.at[index, 'content_emb'] = new_content_emb

            # # Update the row in the table
            # update_query = """
            #     UPDATE tbl_ai_a2p_content
            #     SET content_emb = %s
            #     WHERE id = %s;
            # """
            # with engine.connect() as conn:
            #     conn.execute(
            #         update_query, (new_content_emb.tobytes(), row['id']))
        else:
            new_content_emb = bytes_to_np_vector(content_emb)
            manual_emb_df.at[index, 'content_emb'] = new_content_emb

    # Renaming and type conversion
    manual_emb_df.rename(
        columns={'operator_id': 'operator_name', 'id': 'record_id'}, inplace=True)
    manual_emb_df['operator_name'] = manual_emb_df['operator_name'].astype(int)

    # Filtering based on 'manual_train' flag
    manual_emb_df = manual_emb_df[manual_emb_df['operator_name'].isin(
        operator_df[operator_df['manual_train'] == 1]['operator_name'])]
    manual_emb_df['is_in_db'] = 1

    # Append default values
    zero_vector = np.zeros(384, dtype=np.float32)
    default_row = {
        'record_id': [0],
        'operator_name': [10000],
        'content': ['Default Content'],
        'content_emb': [zero_vector],
        'is_in_db': [1]
    }
    default_row_df = pd.DataFrame(default_row)
    manual_emb_df = pd.concat(
        [manual_emb_df, default_row_df], ignore_index=True)

    return manual_emb_df


def get_auto_emb_df_from_db(operator_df):
    auto_emb_query = "SELECT * FROM operator_embeddings;"
    auto_emb_df = pd.read_sql_query(auto_emb_query, engine)

    auto_emb_df['operator_name'] = auto_emb_df['operator_name'].astype(int)
    auto_emb_df['content_emb'] = auto_emb_df['content_emb'].apply(
        lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else np.nan
    )

    auto_emb_df = auto_emb_df[auto_emb_df['operator_name'].isin(
        operator_df[operator_df['auto_train'] == 1]['operator_name'])]
    auto_emb_df['is_in_db'] = 1

    return auto_emb_df


def emb_df_from_auto_manual_emb_dfs(auto_emb_df, manual_emb_df):

    emb_df = pd.concat([df.assign(source=name) for df, name in zip(
        [auto_emb_df, manual_emb_df], ['auto', 'manual'])])

    return emb_df


def create_db_embeddings(emb_df, operator_df):

    emb_op_df = pd.merge(emb_df[['record_id', 'operator_name', 'content',
                         'content_emb']], operator_df, on='operator_name', how='inner')

    db_embeddings = np.vstack(emb_op_df['content_emb'].values)
    block_arr = np.vstack(emb_op_df['block_score'].values).flatten()
    operator_order_arr = emb_op_df['operator_name'].to_numpy()
    return db_embeddings, block_arr, operator_order_arr


def initialize_startup_variables():

    global db_embeddings, block_arr, operator_order_arr, operator_detect_dict, operator_auto_train_dict, manual_emb_df, auto_emb_df, emb_df, operator_df

    operator_df = make_operator_df()

    operator_detect_dict, operator_auto_train_dict, operator_manual_train_dict = make_dicts_from_operator_df(
        operator_df)

    manual_emb_df = get_manual_emb_df_from_db(operator_df)

    auto_emb_df = get_auto_emb_df_from_db(operator_df)

    emb_df = emb_df_from_auto_manual_emb_dfs(auto_emb_df, manual_emb_df)

    db_embeddings, block_arr, operator_order_arr = create_db_embeddings(
        emb_df, operator_df)

    return db_embeddings, block_arr, operator_order_arr, operator_detect_dict, operator_auto_train_dict, manual_emb_df, auto_emb_df, emb_df, operator_df


# Utility Functions

def bytes_to_np_vector(byte_data):
    return np.frombuffer(byte_data, dtype=np.float32)

# Test