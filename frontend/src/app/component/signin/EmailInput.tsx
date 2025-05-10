interface EmailInputProps {
  email: string;
  setEmail: (email: string) => void;
}

const EmailInput = ({ email, setEmail }: EmailInputProps) => {
  return (
    <div>
      <input
        type="email"
        placeholder="Enter your email"
        className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
    </div>
  );
};

export default EmailInput;
