export function formatFileSize(bytes) {
  return `${(bytes / 1024).toFixed(1)} KB`
}
