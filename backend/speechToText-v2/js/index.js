//convert Float32Array to Int16Array
export function convertToInt16(inputData) {
  const pcm16 = new Int16Array(inputData.length);
  for (let i = 0; i < inputData.length; i++) {
    pcm16[i] = inputData[i] * 32767;
  }
  return pcm16;
}
//convert Float32Array to blob
export function convertToBlob(inputData) {
  const pcm16 = convertToInt16(inputData);
  const blob = new Blob([pcm16], { type: "audio/pcm" });
  return blob;
}
