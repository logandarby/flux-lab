import { Link, useRouter } from "@tanstack/react-router";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardAction,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

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
    id: "divergence",
    title: "Divergence Test",
    description:
      "Tests velocity field divergence calculation for incompressible flow.",
    status: "planned",
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

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Fluid Simulation Tests
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Collection of WebGPU-based fluid simulation tests and experiments.
            Each test focuses on a specific aspect of computational fluid
            dynamics.
          </p>
        </div>

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
                    {test.status === "experimental" ? "In Progress" : "Planned"}
                  </Button>
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-12 text-center">
          <Button variant="link" asChild>
            <Link to="/">← Back to Home</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

export default TestsPage;
