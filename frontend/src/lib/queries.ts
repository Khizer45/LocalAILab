import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { ModelsResponse, SettingsResponse } from "../api/types";

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: () => api<ModelsResponse>("/api/models"), refetchInterval: 30_000 });
}

export function useSettings() {
  return useQuery({ queryKey: ["settings"], queryFn: () => api<SettingsResponse>("/api/settings") });
}

export interface AuthConfig {
  auth_enabled: boolean;
  registration_open: boolean;
  has_users: boolean;
}

export interface Me {
  id: number;
  username: string;
  is_admin: boolean;
  auth_enabled: boolean;
}
