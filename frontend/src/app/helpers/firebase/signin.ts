import { auth } from "@/app/helpers/firebase/firebase";
import { signInWithEmailAndPassword } from "firebase/auth";

export const handleSignIn = async (e: any, email: string, password: string) => {
  e.preventDefault();
  try {
    await signInWithEmailAndPassword(auth, email, password);
    return true;
  } catch (error) {
    alert("Mật khẩu hoặc email không đúng");
    return false;
  }
};
