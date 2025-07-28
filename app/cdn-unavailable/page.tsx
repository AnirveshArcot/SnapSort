export default function CDNUnavailable() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-center">
      <h1 className="text-3xl font-bold">CDN Not Mounted</h1>
      <p className="mt-4 text-lg text-gray-600">
        The image storage system is not available. Please try again later.
      </p>
    </div>
  );
}
