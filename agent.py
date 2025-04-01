import os
import yaml
from crewai import Crew, Task, Process, Agent, LLM


llm = LLM(model=os.environ.get("LLM_MODEL"), temperature=0, api_key=os.environ.get("LLM_API_KEY"))


class CrewAgent:
    """Generic Crew agent"""

    def __init__(self):

        def open_yaml(nome: str) -> dict:
            with open(nome, 'r') as f:
                try:
                    return yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)

        self.agents_yaml = open_yaml('config/agents.yaml')
        self.tasks_yaml = open_yaml('config/tasks.yaml')
        self.agents_dict = {}
        self.tasks_list = []

    def new_agent(self, agent_name: str, tools: list=[]) -> Agent:
        self.agents_dict[agent_name] = Agent(
                                            config=self.agents_yaml[agent_name],
                                            tools=tools,
                                            verbose=True,
                                            allow_delegation=False,
                                            llm=llm
                                        )
                
    def new_task(self, task_name: str, agent: Agent) -> Task:
        self.tasks_list += [Task(
                                config=self.tasks_yaml[task_name],
                                agent=agent,
                            )]
    
    def crew(self) -> Crew:
        return Crew(
            agents=list(self.agents_dict.values()),
            tasks=self.tasks_list,
            verbose=True,
            process=Process.sequential,
        )
    
    if __name__ == "__main__":
        from agent import CrewAgent
        agent = CrewAgent()
        agent.new_agent(agent_name='monitor_agent')
        agent.new_task('chat_task', agent=agent.agents_dict['monitor_agent'])
        print(agent.tasks_yaml['chat_task'])