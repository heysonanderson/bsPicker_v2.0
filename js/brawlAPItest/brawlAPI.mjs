const API_KEY = 'api ne budet';
const url = `https://api.brawlstars.com/v1/rankings/global/players`;


import fetch from 'node-fetch'
import fs from 'fs'

async function getPlayerData() {
  try {
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${API_KEY}`
      }
    });
    if (!response.ok) throw new Error(`Ошибка: ${response.status}`);

    const data = await response.json();


    fs.writeFileSync('./aboba/global_top_players.json', JSON.stringify(data, null, 2));


    console.log(data);
  } catch (error) {
    console.error('Ошибка:', error.message);
  }
}

getPlayerData();

