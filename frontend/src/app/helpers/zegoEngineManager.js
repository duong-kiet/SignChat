import { ZegoExpressEngine } from "zego-express-engine-webrtc";

export const createZegoEngine = (appID, server, config) => {
  const zg = new ZegoExpressEngine(appID, server);
  zg.setLogConfig(config);
  return zg;
};

export const loginToRoom = async (zg, roomID, token, userID, userName) => {
  const result = await zg.loginRoom(
    roomID,
    token,
    { userID, userName },
    { userUpdate: true }
  );
  return result;
};

export const createStream = async (zg) => {
  const localStream = await zg.createStream({
    camera: { audio: true, video: true },
  });
  return localStream;
};
