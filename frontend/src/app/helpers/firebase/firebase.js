import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getDatabase } from "firebase/database";

// Your web app's Firebase configuration

const firebaseConfig = {
  apiKey: "AIzaSyDXbP8l-LDIBc21dyugttwANnsSxD7dk1g",
  authDomain: "signmeet-aacac.firebaseapp.com",
  databaseURL: "https://signmeet-aacac-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "signmeet-aacac",
  storageBucket: "signmeet-aacac.firebasestorage.app",
  messagingSenderId: "837049587631",
  appId: "1:837049587631:web:482765c4ae82710c26015a"
};



const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getDatabase();

export { auth, db };
