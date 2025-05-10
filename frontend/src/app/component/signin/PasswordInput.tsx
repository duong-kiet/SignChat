interface PasswordInputProps {
  password: string;
  setPassword: (password: string) => void;
}

const PasswordInput = ({ password, setPassword }: PasswordInputProps) => {
  return (
    <div className="relative">
      <input
        type="password"
        placeholder="Enter your password"
        className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <button className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
        Forgot?
      </button>
    </div>
  );
};

export default PasswordInput;
