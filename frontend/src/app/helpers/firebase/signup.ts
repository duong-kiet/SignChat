import { auth, db } from "@/app/helpers/firebase/firebase";
import { ref, set } from "firebase/database";
import { createUserWithEmailAndPassword } from "firebase/auth";

export const handleSignUp = async (
  e: any,
  email: string,
  password: string,
  firstName: string,
  lastName: string,
  username: string
): Promise<boolean> => {
  e.preventDefault();
  try {
    const userCredential = await createUserWithEmailAndPassword(
      auth,
      email,
      password
    );
    const user = userCredential.user;
    const userData = {
      email: user.email,
      firstName,
      lastName,
      username,
    };
    await set(ref(db, `users/${user.uid}`), userData);
    alert("User created successfully");
    return true;
  } catch (error) {
    console.log(error);
    return false;
  }
};
