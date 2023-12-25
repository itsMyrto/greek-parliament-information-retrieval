import lsi
import search_engine

print(""" 1.) Search Query in the greek parliament dataset \n 4.) Perform LSI   """)

ans = input('Enter 1 or 4: ')

if ans == "1":
    search_engine.search_query()
elif ans == "4":
    lsi_ = lsi.LSI()
else:
    print("I said 1 or 4, not fcking ", ans)
