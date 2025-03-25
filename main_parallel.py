from mpi4py import MPI
from collections import defaultdict
import heapq
import ujson as json

SMALL_FILE_PATH = 'mastodon-106k.ndjson'
MIDIUM_FILE_PATH = 'mastodon-16m.ndjson'
LARGE_FILE_PATH = 'mastodon-144g.ndjson'

BUFFER_SIZE = 1024 * 1024 * 64
def merge_dicts(a, b, dtype):
    # Iterate over each key-value pair in dictionary b
    for key, value in b.items():
        # a.get(key, 0) gets the value from a for the given key, or 0 if key not present
        a[key] = a.get(key, 0) + value
    return a

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #record the time
    start_time = MPI.Wtime()

    # we always choose rank 0 as scheduler
    file_size = None
    if rank == 0:
        with open(SMALL_FILE_PATH, "rb") as f:
            f.seek(0, 2)  # move to the end of the file
            file_size = f.tell()
        print("file size:{}",file_size)
    
    if size>1:
        file_size = comm.bcast(file_size, root=0)

    # calculate chunks size of each rank
    chunk_size = file_size // size
    start_offset = rank * chunk_size
    # give all left byte for the last rank
    end_offset = (rank + 1) * chunk_size if rank != size - 1 else file_size

    # local_data = []
    sentiment_by_hour = defaultdict(float)
    sentiment_by_user = defaultdict(float)
    with open(SMALL_FILE_PATH, "r", encoding="utf-8") as f:
        # If not starting from the beginning of the file, move to the start of a full line
        if start_offset > 0:
            f.seek(start_offset)
            f.readline()  # Skip possibly incomplete line

        while True:
            pos = f.tell()
            # Stop if current position exceeds the range for this process
            if pos >= end_offset:
                break
            line = f.readline()
            if not line:
                break  # Reached end of file

            # Process each line of data
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                doc = data.get("doc", {})
                sentiment = doc.get("sentiment", None)
                user_info = doc.get("account", {})
                user_id = user_info.get("id")
                username = user_info.get("username")
                time = doc.get("createdAt", None)

                if user_id and username and sentiment is not None:
                    sentiment_by_user[(user_id, username)] += sentiment
                if time and sentiment is not None:
                    hour = time[:13]  # Extract hour (YYYY-MM-DDTHH)
                    sentiment_by_hour[hour] += sentiment
            except json.JSONDecodeError:
                continue  # Skip lines with format errors

    print(f"Rank {rank} processed {count} records.")

    # calculate
    avg_hour_sentiment =  sentiment_by_hour
    local_avg_hours_sentiment = [(hours,score) for hours,score in avg_hour_sentiment.items()]
    avg_user_sentiment = sentiment_by_user
    local_avg_user_sentiment = [(user_id, username, sentiment) for (user_id, username), sentiment in avg_user_sentiment.items()]

    local_avg_hours_sentiment = dict(avg_hour_sentiment)
    local_avg_user_sentiment = dict(avg_user_sentiment)

    dict_op = MPI.Op.Create(merge_dicts, commute=True)

    # gather the result into rank0
    if size>1:
        gathered_hours = comm.reduce(local_avg_hours_sentiment, op=dict_op, root=0)
        gathered_users = comm.reduce(local_avg_user_sentiment, op=dict_op, root=0)
        
        global_hours_sentiment = gathered_hours
        global_users_sentiment = gathered_users
    else:
        global_hours_sentiment = sentiment_by_hour
        global_users_sentiment = sentiment_by_user

    if rank == 0:

        print(f"combined users size: {len(global_users_sentiment)}")
        
        global_users_top5 = heapq.nlargest(5, ((s, uid, uname) for (uid, uname), s in global_users_sentiment.items()))
        global_users_bottom5 = heapq.nsmallest(5, ((s, uid, uname) for (uid, uname), s in global_users_sentiment.items()))
        
        global_hours_top5 = heapq.nlargest(5, ((score, hour) for hour, score in global_hours_sentiment.items()))
        global_hours_bottom5 = heapq.nsmallest(5, ((score, hour) for hour, score in global_hours_sentiment.items()))
        

        # record the end time
        end_time = MPI.Wtime()
        print(f"\n Finish, takes {end_time - start_time:.2f} s")
        
        print("\n Happest 5 hours:")
        for score, hours in global_hours_top5:
            print(f"{hours} -> {score:.2f}")

        print("\n🔹 Saddest 5 hours:")
        for score, hours in global_hours_bottom5:
            print(f"{hours} -> {score:.2f}")

        print("\n🔹 Happest 5 users:")
        for sentiment, user_id, username in global_users_top5:
            print(f"{username} (ID: {user_id}) -> {sentiment:.2f}")

        print("\n🔹 Saddest 5 users:")
        for sentiment, user_id, username in global_users_bottom5:
            print(f"{username} (ID: {user_id}) -> {sentiment:.2f}")


