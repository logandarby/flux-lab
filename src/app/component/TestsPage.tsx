import { Link, useRouter } from "@tanstack/react-router";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardAction,
} from "@/shared/ui/card";
import { Button } from "@/shared/ui/button";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/shared/utils/utils";

interface TestInfo {
  id: string;
  title: string;
  description: string;
  status: "stable" | "experimental" | "planned";
}

const tests: TestInfo[] = [
  {
    id: "advection",
    title: "Advection Test",
    description:
      "Tests fluid velocity advection with manual bilinear interpolation and rg32float textures.",
    status: "stable",
  },
  {
    id: "diffusion",
    title: "Diffusion Test",
    description:
      "Tests fluid property diffusion using compute shaders with texture ping-ponging.",
    status: "stable",
  },
];

function TestsPage() {
  const router = useRouter();

  const getStatusColor = (status: TestInfo["status"]) => {
    switch (status) {
      case "stable":
        return "bg-green-100 text-green-800 border-green-200";
      case "experimental":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "planned":
        return "bg-gray-100 text-gray-800 border-gray-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const handleTestNavigation = (testId: string) => {
    router.navigate({ to: "/tests/$testId", params: { testId } });
  };

  // Header component with back button
  const TestsHeader = () => (
    <div className="border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" asChild>
            <Link to="/" className="flex items-center gap-2">
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Link>
          </Button>
          <div className="h-6 w-px bg-border" />
          <h1 className="text-lg font-semibold text-gray-900">
            Fluid Simulation Tests
          </h1>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <TestsHeader />
      <div className="container mx-auto py-8">
        <div className="max-w-4xl mx-auto px-6">
          <div className="grid gap-6 md:grid-cols-2">
            {tests.map((test) => (
              <Card key={test.id}>
                <CardHeader>
                  <CardTitle>{test.title}</CardTitle>
                  <CardAction>
                    <span
                      className={cn(
                        "px-3 py-1 rounded-full text-xs font-medium border",
                        getStatusColor(test.status)
                      )}
                    >
                      {test.status}
                    </span>
                  </CardAction>
                </CardHeader>

                <CardContent>
                  <CardDescription className="mb-6">
                    {test.description}
                  </CardDescription>

                  {test.status === "stable" ? (
                    <Button
                      onClick={() => handleTestNavigation(test.id)}
                      className="w-full"
                    >
                      Launch Test
                    </Button>
                  ) : (
                    <Button disabled variant="secondary" className="w-full">
                      {test.status === "experimental"
                        ? "In Progress"
                        : "Planned"}
                    </Button>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default TestsPage;
