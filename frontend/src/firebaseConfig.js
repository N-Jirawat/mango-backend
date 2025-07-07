import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyB4qrpqEKkYX_peX17H3hxQjBIKGdJmeUI",
  authDomain: "mango-e6bd0.firebaseapp.com",
  databaseURL: "https://mango-e6bd0-default-rtdb.firebaseio.com",
  projectId: "mango-e6bd0",
  storageBucket: "mango-e6bd0.firebasestorage.app",
  messagingSenderId: "228987366685",
  appId: "1:228987366685:web:dcc3c791efc448a16d45f4"
};

// ✅ เริ่มต้น Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);
const storage = getStorage(app);

export { app, db, auth, storage };
