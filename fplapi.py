from fpl import FPL

async def get_player_data():
    fpl = FPL()
    await fpl.login()  # Optional, if you need to access authenticated data
    elements = await fpl.get_elements()
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(elements)
    print(df.head())
    return df

# Run the async function (requires `import asyncio`)
import asyncio
asyncio.run(get_player_data())