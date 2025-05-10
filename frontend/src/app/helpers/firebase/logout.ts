import { auth } from "@/app/helpers/firebase/firebase";
import { signOut } from "firebase/auth";

export const handleSignOut = async () => {
  signOut(auth)
    .then(() => {
      window.location.href = "/signin";
    })
    .catch((error) => {
      alert("Đăng xuất thất bại");
    });
};
