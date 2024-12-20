import psycopg2
from psycopg2 import Error

def connect_to_postgres():
    try:
        # Connection parameters - replace these with your actual database details
        connection = psycopg2.connect(
            host="localhost",
            database="impact_analyzer",
            user="postgres",
            password="postgres",
            port="5432"
        )
        
        # Create a cursor object to execute queries
        cursor = connection.cursor()
        
        # Print PostgreSQL details
        print("PostgreSQL server information:")
        print(connection.get_dsn_parameters())
        
        # Execute a test query
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        print("You are connected to - ", record)
        
        return connection, cursor
        
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL:", error)
        return None, None

def close_connection(connection, cursor):
    if cursor:
        cursor.close()
    if connection:
        connection.close()
        print("PostgreSQL connection is closed")

def execute_query(query, params=None):
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(
            host="localhost",
            database="impact_analyzer",
            user="postgres",
            password="postgres",
            port="5432"
        )
        cursor = connection.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        # Check if the query is a SELECT statement
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            return column_names, results
        else:
            connection.commit()
            return None, cursor.rowcount
            
    except (Exception, Error) as error:
        print("Error while executing query:", error)
        return None, None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def get_server_account_mappings():
    query = """
    SELECT DISTINCT
        v.account_id,
        v.node_name as server_name
    FROM 
        vm_list v
    INNER JOIN server_list s ON v.node_name = s.server_name
    INNER JOIN power_metrics p ON v.node_name = p.server_name
    ORDER BY 
        v.account_id;
    """
    
    columns, results = execute_query(query)
    return columns, results

if __name__ == "__main__":
    # Test the connection
    conn, cur = connect_to_postgres()
    if conn:
        close_connection(conn, cur)

    # Example query to list all tables in the database
    query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """
    
    print("Listing all tables in the database:")
    columns, tables = execute_query(query)
    if tables:
        for table in tables:
            print(f"- {table[0]}")

    print("\nFetching server-account mappings:")
    columns, mappings = get_server_account_mappings()
    if mappings:
        print("\nColumns:", columns)
        print("\nServer-Account Mappings:")
        for row in mappings:
            print(f"Account ID: {row[0]} | Server Name: {row[1]}")
    else:
        print("No mappings found or an error occurred")
