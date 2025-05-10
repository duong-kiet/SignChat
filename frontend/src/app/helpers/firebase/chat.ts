import { auth, db } from "./firebase";
import { push, ref, set, onChildAdded, remove, get } from "firebase/database";

export const sendMessage = async (content: string) => {
  const userId = auth.currentUser?.uid;
  let userName = "";

  // Fetch user name
  const userRef = ref(db, "users");
  try {
    const snapshot = await get(userRef);
    snapshot.forEach((childSnapshot) => {
      if (childSnapshot.val()["email"] === auth.currentUser?.email) {
        userName = childSnapshot.val()["lastName"];
      }
    });
  } catch (error) {
    console.error("Error fetching user name:", error);
  }

  if (userId && content.trim()) {
    set(push(ref(db, "chats")), {
      content,
      userName,
      userId,
      timestamp: Date.now(),
    });
  }

  return { content, userName, userId };
};

export const clearChat = () => {
  const chatList = ref(db, "chats");
  remove(chatList);
};

export const subscribeToMessages = (callback: (message: any) => void) => {
  const chatList = ref(db, "chats");
  return onChildAdded(chatList, async (data) => {
    const { content, userId, userName } = data.val();
    if (userId !== auth.currentUser?.uid) {
      callback({ text: content, sender: userName, userId });
    }
  });
};
