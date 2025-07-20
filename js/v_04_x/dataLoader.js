export async function getData(url, name) {
    const response = await fetch(url + name);
    const csvText = await response.text();
    return Papa.parse(csvText, { header: true, skipEmptyLines: true }).data;
}