const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  if (!BASE_URL) {
    throw new Error("Backend URL is not configured");
  }

  const response = await fetch(`${BASE_URL}${path}`, options);

  if (!response.ok) {
    throw new Error(`Request failed (${response.status})`);
  }

  return response.status === 204 ? null : response.json();
}

export const getHealth = async () => {
  if (!BASE_URL) return { ok: false };

  try {
    await request("/health");
    return { ok: true };
  } catch {
    return { ok: false };
  }
};

export const uploadStudy = (file) => {
  const body = new FormData();
  body.append("file", file);

  return request("/studies", {
    method: "POST",
    body,
  });
};

export const getStudies = () => request("/studies");

export const getStudyResult = (id) =>
  request(`/studies/${encodeURIComponent(id)}/results`);
