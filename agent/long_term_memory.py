from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

namespace = ("user", "rahul")

store.put(
    namespace,
    "name",
    {
        "name": "Rahul"
    }
)

memory = store.get(
    namespace,
    "name"
)

print(memory)